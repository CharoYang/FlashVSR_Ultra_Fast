import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
import torchvision
import cv2
from tqdm import tqdm
from einops import rearrange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import uuid
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ====== FlashVSR modules ======
from models.model_manager import ModelManager
from models.TCDecoder import build_tcdecoder
from models.utils import Buffer_LQ4x_Proj, clean_vram
from models import wan_video_dit
from pipelines.flashvsr_full import FlashVSRFullPipeline
from pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline

# ==============================================================
#                      Utility Functions
# ==============================================================

def get_device_list():
    """Return list of available devices."""
    devs = ["auto"]
    try:
        if torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs

def get_gpu_memory_info(device: str) -> Tuple[float, float]:
    """Get GPU memory info (used, total) in GB."""
    if not device.startswith("cuda:"):
        return 0.0, 0.0
    try:
        idx = int(device.split(":")[1])
        torch.cuda.set_device(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        allocated = torch.cuda.memory_allocated(idx) / (1024**3)
        used = reserved  # 使用reserved memory作为使用量
        free = total - used
        return used, total
    except Exception as e:
        log(f"Error getting GPU memory info: {e}", "warning")
        return 0.0, 0.0

def get_available_memory_gb(device: str) -> float:
    """Get available GPU memory in GB."""
    used, total = get_gpu_memory_info(device)
    return total - used

def estimate_tile_memory(tile_size: int, num_frames: int, scale: int, dtype_size: int = 2) -> float:
    """Estimate memory needed for processing one tile in GB.
    
    Args:
        tile_size: Tile size in pixels
        num_frames: Number of frames
        scale: Upscale factor
        dtype_size: Size of dtype in bytes (2 for fp16/bf16, 4 for fp32)
    """
    # 更准确的显存估算
    # 输入：tile_size^2 * num_frames * 3 * dtype_size
    input_size = tile_size * tile_size * num_frames * 3 * dtype_size / (1024**3)
    
    # 输出：tile_size^2 * scale^2 * num_frames * 3 * dtype_size
    output_size = (tile_size * scale) * (tile_size * scale) * num_frames * 3 * dtype_size / (1024**3)
    
    # 中间激活：由于使用tiled处理，实际激活显存较小，约5-8x输入（更激进的估算）
    # 同时处理多个tile时，某些激活可以共享
    intermediate_size = input_size * 6  # 从12倍降低到6倍，更符合实际
    
    # 添加一些额外开销（梯度缓冲区等，虽然inference不需要，但框架可能有保留）
    overhead = 0.5  # 0.5GB额外开销
    
    return input_size + intermediate_size + output_size + overhead

def log(message: str, message_type: str = 'normal'):
    """Colored logging for console output."""
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    print(message)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def tensor2video(frames: torch.Tensor):
    """Convert tensor (B,C,F,H,W) to normalized video tensor (F,H,W,C) - 与 nodes.py 一致"""
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):
    """Return largest (8n+1) less than or equal to n."""
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    """Compute scaled and target dimensions aligned to multiple."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid input size")
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int):
    """Upscale and center-crop a tensor frame."""
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    sW, sH = w0 * scale, h0 * scale
    upscaled = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped = upscaled[:, :, t:t + tH, l:l + tW]
    return cropped.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """Prepare video tensor by upscaling and padding."""
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F_ = largest_8n1_leq(num_frames_with_padding)
    if F_ == 0:
        raise RuntimeError(f"Not enough frames after padding: {num_frames_with_padding}")

    frames = []
    for i in tqdm(range(F_), desc="Preparing frames", ncols=80):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale, tW, tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to('cpu').to(dtype)
        frames.append(tensor_out)

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F_

def calculate_tile_coords(height, width, tile_size, overlap):
    """Calculate tile coordinates for patch-based inference."""
    coords = []
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap):
    """Create blending mask for overlapping tiles."""
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask

def pad_or_crop_video(frames):
    """Ensure temporal and spatial alignment (8n+1, 32x multiple)."""
    T, C, H, W = frames.shape
    aligned_F = largest_8n1_leq(T)
    if aligned_F < T:
        frames = frames[:aligned_F]
    elif aligned_F > T:
        pad = frames[-1:].repeat(aligned_F - T, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
    new_H = (H // 32) * 32
    new_W = (W // 32) * 32
    frames = frames[:, :, :new_H, :new_W]
    return frames

def save_video(frames, path, fps=30):
    """Save tensor video frames to MP4 - frames is (F, H, W, C) format."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    # frames is (F, H, W, C), convert to numpy
    frames_np = (frames.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
    h, w = frames_np.shape[1:3]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames_np:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

def read_video_to_tensor(video_path):
    """Read video and convert to (N, H, W, C) format for prepare_input_tensor.

    Primary backend: torchvision.io.read_video (ffmpeg-based)
    Fallback: OpenCV VideoCapture (handles many H.264/H.265 cases in containers)
    """
    try:
        vr = torchvision.io.read_video(video_path, pts_unit='sec')[0]
        if vr.numel() > 0 and vr.shape[0] > 0:
            vr = vr.permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W)
            return vr.permute(0, 2, 3, 1)  # (N, H, W, C)
        else:
            log(f"[read_video] torchvision returned empty tensor for: {video_path}", "warning")
    except Exception as e:
        log(f"[read_video] torchvision read_video failed: {e}", "warning")

    # Fallback to OpenCV
    log("[read_video] Falling back to OpenCV VideoCapture...", "warning")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame_rgb).float() / 255.0)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"[read_video] No frames decoded from: {video_path}")
    vr = torch.stack(frames, dim=0)  # (N,H,W,C)
    return vr

def split_video_by_frames(frames: torch.Tensor, num_gpus: int, overlap: int = 10):
    N = frames.shape[0]
    segment_size = N // num_gpus
    segments = []
    for i in range(num_gpus):
        start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
        end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
        segments.append((start_idx, end_idx))  # 仅返回区间
    return segments

def merge_video_segments(segments: List[Tuple[int, int, torch.Tensor]], original_length: int) -> torch.Tensor:
    """Merge processed video segments back into a single video.
    
    Args:
        segments: List of (start_idx, end_idx, processed_segment) tuples
        original_length: Original number of frames
    
    Returns:
        Merged video tensor (F, H, W, C)
    """
    if not segments:
        raise ValueError("No segments to merge")
    
    segments = sorted(segments, key=lambda x: x[0])
    
    # 简单的合并策略：直接连接segments，处理overlap区域
    merged_parts = []
    
    for i, (start_idx, end_idx, segment) in enumerate(segments):
        segment_frames = segment.shape[0]
        
        if i == 0:
            # 第一个segment：保留全部，但需要根据原始长度调整
            # 计算应该保留多少帧
            if len(segments) == 1:
                # 只有一个segment，直接裁剪
                merged_parts.append(segment[:original_length])
            else:
                # 保留到下一个segment的start_idx（考虑overlap）
                next_start = segments[i+1][0] if i+1 < len(segments) else original_length
                keep_frames = min(segment_frames, next_start - start_idx)
                merged_parts.append(segment[:keep_frames])
        else:
            # 后续segments：跳过overlap部分
            prev_end = segments[i-1][1]
            overlap = max(0, start_idx - prev_end)
            
            # 计算当前segment应该从哪一帧开始
            segment_start_frame = min(overlap, segment_frames)
            
            # 计算应该保留到哪一帧
            if i == len(segments) - 1:
                # 最后一个segment：保留到original_length
                frames_needed = original_length - (sum(p.shape[0] for p in merged_parts) + segment_start_frame)
                keep_frames = min(segment_frames - segment_start_frame, frames_needed)
                if keep_frames > 0:
                    merged_parts.append(segment[segment_start_frame:segment_start_frame + keep_frames])
            else:
                # 中间segments：保留到下一个segment的start_idx
                next_start = segments[i+1][0]
                current_merged_length = sum(p.shape[0] for p in merged_parts)
                frames_needed = next_start - current_merged_length - segment_start_frame
                keep_frames = min(segment_frames - segment_start_frame, frames_needed)
                if keep_frames > 0:
                    merged_parts.append(segment[segment_start_frame:segment_start_frame + keep_frames])
    
    if not merged_parts:
        raise ValueError("Failed to merge segments")
    
    merged = torch.cat(merged_parts, dim=0)
    
    # 确保长度正确
    if merged.shape[0] > original_length:
        merged = merged[:original_length]
    elif merged.shape[0] < original_length:
        # 如果长度不够，重复最后一帧
        last_frame = merged[-1:].repeat(original_length - merged.shape[0], 1, 1, 1)
        merged = torch.cat([merged, last_frame], dim=0)
    
    return merged

def run_inference_multi_gpu(frames: torch.Tensor, devices: List[str], args):
    """Run inference using multiple GPUs in parallel."""
    process_start = time.time()
    num_gpus = len(devices)
    if num_gpus == 0:
        raise ValueError("No GPUs specified for multi-GPU processing")
    
    log(f"[Multi-GPU] Processing video with {num_gpus} GPUs", "info")
    
    # 将视频分割成segments
    segments = split_video_by_frames(frames, num_gpus, overlap=10)
    log(f"[Multi-GPU] Split video into {len(segments)} segments", "info")
    for i, (start, end) in enumerate(segments):
        log(f"[Multi-GPU] Segment {i}: frames {start}-{end} ({end-start} frames) -> GPU {devices[i % num_gpus]}", "info")
    
    # 使用多进程处理
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []
    
    # 准备参数字典
    args_dict = vars(args)
    
    # 启动worker进程
    log(f"[Multi-GPU] Launching {num_gpus} worker processes...", "info")
    for i, (start_idx, end_idx) in enumerate(segments):
        device = devices[i % num_gpus]
        log(f"[Multi-GPU] Starting worker {i} on {device}", "info")
        p = ctx.Process(
            target=_worker_process,
            args=(i, device, args.input, start_idx, end_idx, args_dict, result_queue)
        )
        p.start()
        processes.append(p)
        log(f"[Multi-GPU] Worker {i} process started (PID: {p.pid})", "info")
    
    log(f"[Multi-GPU] All workers started. Waiting for results...", "info")
    log(f"[Multi-GPU] Monitoring progress (checking every 2 seconds)...", "info")
    
    # 收集结果（添加进度显示）
    results = {}
    completed = 0
    total = num_gpus
    last_progress_time = time.time()
    
    while completed < total:
        try:
            # 检查是否有进度消息
            if not result_queue.empty():
                result = result_queue.get(timeout=0.1)
                
                if result.get('type') == 'progress':
                    # 显示进度消息
                    log(f"[Worker {result['worker_id']}@{result['device']}] {result['stage']}: {result['message']}", "info")
                    last_progress_time = time.time()
                elif 'success' in result:
                    # 这是最终结果
                    completed += 1
                    if result['success']:
                        results[result['worker_id']] = {
                            'start_idx': result['start_idx'],
                            'end_idx': result['end_idx'],
                            'path': result['path']
                        }
                        log(f"[Multi-GPU] Worker {result['worker_id']} completed ({completed}/{total})", "finish")
                    else:
                        log(f"[Multi-GPU] Worker {result['worker_id']} failed: {result['error']}", "error")
                        raise RuntimeError(f"Worker {result['worker_id']} failed")
            else:
                # 如果没有消息，显示等待状态
                elapsed = time.time() - last_progress_time
                if elapsed > 5:
                    log(f"[Multi-GPU] Still waiting... ({completed}/{total} completed, {elapsed:.1f}s since last update)", "info")
                    last_progress_time = time.time()
                    # 检查进程是否还活着
                    alive_count = sum(1 for p in processes if p.is_alive())
                    log(f"[Multi-GPU] {alive_count}/{len(processes)} processes are alive", "info")
                time.sleep(0.5)
        except:
            time.sleep(0.5)
            continue
    
    # 等待所有进程完成
    log(f"[Multi-GPU] All workers finished. Waiting for processes to exit...", "info")
    for i, p in enumerate(processes):
        p.join(timeout=30)
        if p.exitcode != 0:
            log(f"[Multi-GPU] Process {i} exited with code {p.exitcode}", "error")
        else:
            log(f"[Multi-GPU] Process {i} exited successfully", "info")
    
    # 合并segments（从临时文件读取并清理）
    log(f"[Multi-GPU] Merging {len(results)} segments...", "info")
    segment_list = []
    for i in sorted(results.keys()):
        path = results[i]['path']
        out = torch.load(path, map_location='cpu')
        segment_list.append((results[i]['start_idx'], results[i]['end_idx'], out))
        try:
            os.remove(path)
        except Exception:
            pass

    merged_output = merge_video_segments(segment_list, frames.shape[0])
    process_time = time.time() - process_start
    log(f"[Multi-GPU] Successfully processed and merged {num_gpus} segments", "finish")
    log(f"[Multi-GPU] Processing time: {format_duration(process_time)}", "finish")
    return merged_output

def _worker_process(worker_id: int, device: str, video_path: str,
                   start_idx: int, end_idx: int, args_dict: dict, result_queue: mp.Queue):
    """Worker process for multi-GPU processing (separate function to avoid import issues)."""
    try:
        import sys
        sys.stdout.flush()  # 确保输出立即显示
        sys.stderr.flush()
        
        # 添加进度报告
        def report_progress(stage, message):
            """向主进程报告进度"""
            try:
                result_queue.put({
                    'worker_id': worker_id,
                    'type': 'progress',
                    'stage': stage,
                    'message': message,
                    'device': device
                }, block=False)
            except:
                pass
            print(f"[Worker {worker_id}@{device}] {stage}: {message}", flush=True)
            sys.stdout.flush()
        
        report_progress("INIT", "Worker process started")
        
        # 重新导入必要的模块（在子进程中）
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        sys.path.insert(0, os.path.join(script_dir, "src"))
        
        report_progress("INIT", "Importing modules...")
        
        # 设置当前进程使用的GPU
        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":")[1]))
            report_progress("INIT", f"Set CUDA device to {device}")
        
        # 解析参数
        from argparse import Namespace
        args = Namespace(**args_dict)
        args.device = device
        
        # 导入必要的模块
        report_progress("LOAD", "Loading model modules...")
        from models.utils import clean_vram
        from models import wan_video_dit
        from models.model_manager import ModelManager
        from models.TCDecoder import build_tcdecoder
        from models.utils import Buffer_LQ4x_Proj
        from pipelines.flashvsr_full import FlashVSRFullPipeline
        from pipelines.flashvsr_tiny import FlashVSRTinyPipeline
        from pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline
        
        # 处理attention_mode
        if args.attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
        
        # 初始化pipeline
        report_progress("LOAD", "Initializing pipeline...")
        model_path = args.model_dir
        ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
        tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
        prompt_path = os.path.join(script_dir, "posi_prompt.pth")
        
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(args.precision, torch.bfloat16)
        
        report_progress("LOAD", "Loading model weights...")
        mm = ModelManager(torch_dtype=dtype, device="cpu")
        if args.mode == "full":
            mm.load_models([ckpt_path, vae_path])
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
        else:
            mm.load_models([ckpt_path])
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if args.mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            multi_scale_channels = [512, 256, 128, 128]
            pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
            pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
            pipe.TCDecoder.clean_mem()
        
        report_progress("LOAD", "Loading additional components...")
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
        pipe.denoising_model().LQ_proj_in.to(device)
        pipe.to(device, dtype=dtype)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(prompt_path=prompt_path)
        pipe.load_models_to_device(["dit", "vae"])
        
        report_progress("PROCESS", "Pipeline initialized. Loading video and running inference...")

        # 读取视频并裁剪到该段
        from infer_video import read_video_to_tensor, run_inference
        frames_all = read_video_to_tensor(video_path)
        # Clamp indices to avoid empty or negative-length segments
        N = frames_all.shape[0]
        start = max(0, min(start_idx, N))
        end = max(start, min(end_idx, N))
        if end <= start:
            raise RuntimeError(f"[Worker {worker_id}] Empty segment after clamp: N={N}, start={start_idx}, end={end_idx}")
        frames_segment = frames_all[start:end]
        del frames_all

        # 执行推理
        output = run_inference(pipe, frames_segment, device, dtype, args)

        # 保存到临时文件，返回路径
        tmp_dir = os.path.join('/tmp', 'flashvsr_multigpu')
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"worker_{worker_id}_{uuid.uuid4().hex}.pt")
        torch.save(output, out_path)

        result_queue.put({
            'worker_id': worker_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'path': out_path,
            'success': True
        })

        report_progress("DONE", f"Results saved to {out_path}")

        del pipe, output, frames_segment
        clean_vram()
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker {worker_id} ERROR] {error_msg}", flush=True)
        result_queue.put({
            'worker_id': worker_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'error': error_msg,
            'success': False
        })

# ==============================================================
#                     Padding for Model Input
# ==============================================================

def pad_to_window_multiple(frames: torch.Tensor, window=(2, 8, 8)):
    """Pad tensor so that F/H/W are multiples of given window size."""
    win_t, win_h, win_w = window
    shape = list(frames.shape)
    if len(shape) == 4:  # (F,C,H,W)
        f, c, h, w = shape
        prefix = ()
    elif len(shape) == 5:  # (B,C,F,H,W)
        prefix = (0, 1)
        f, c, h, w = shape[2:]
    else:
        raise ValueError(f"Unexpected input shape: {shape}")

    new_f = math.ceil(f / win_t) * win_t
    new_h = math.ceil(h / win_h) * win_h
    new_w = math.ceil(w / win_w) * win_w
    pad_f, pad_h, pad_w = new_f - f, new_h - h, new_w - w

    if pad_f == 0 and pad_h == 0 and pad_w == 0:
        print(f"[INFO] Already aligned to window multiples ({f},{h},{w})")
        return frames, (f, h, w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))

    if pad_f > 0:
        if len(shape) == 4:
            last = frames[-1:].repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, last], dim=0)
        elif len(shape) == 5:
            last = frames[:, :, -1:].repeat(1, 1, pad_f, 1, 1)
            frames = torch.cat([frames, last], dim=2)

    print(f"[INFO] Padded to ({new_f},{new_h},{new_w}) for window compatibility")
    return frames, (f, h, w)

def pad_frames_auto(input_data, window=(2, 8, 8)):
    """Detects input type (Tensor, dict, or list) and applies padding."""
    if isinstance(input_data, torch.Tensor):
        return pad_to_window_multiple(input_data, window)
    elif isinstance(input_data, dict):
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                padded, orig = pad_to_window_multiple(v, window)
                input_data[k] = padded
                return input_data, orig
    elif isinstance(input_data, (list, tuple)):
        for i in range(len(input_data)):
            if isinstance(input_data[i], torch.Tensor):
                padded, orig = pad_to_window_multiple(input_data[i], window)
                input_data[i] = padded
                return input_data, orig
    raise TypeError(f"[ERROR] Unsupported input type for padding: {type(input_data)}")

# ==============================================================
#                     FlashVSR Pipeline
# ==============================================================

def init_pipeline(mode, device, dtype, model_dir):
    """Initialize FlashVSR pipeline and load model weights."""
    model_path = model_dir
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posi_prompt.pth")

    for p in [ckpt_path, vae_path, lq_path, tcd_path]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing model file: {p}")

    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()

    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"])

    log(f"Pipeline initialized successfully in mode: {mode}", "finish")
    return pipe

def process_tile_batch(pipe, frames, device, dtype, args, tile_batch: List[Tuple[int, int, int, int]], batch_idx: int):
    """Process a batch of tiles and return results."""
    N, H, W, C = frames.shape
    num_aligned_frames = largest_8n1_leq(N + 4) - 4
    
    results = []
    
    for tile_idx, (x1, y1, x2, y2) in enumerate(tile_batch):
        input_tile = frames[:, y1:y2, x1:x2, :]
        
        LQ_tile, th, tw, F = prepare_input_tensor(input_tile, device, scale=args.scale, dtype=dtype)
        if "long" not in args.mode:
            LQ_tile = LQ_tile.to(device)
        
        topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)
        
        with torch.no_grad():
            output_tile = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=args.seed,
                tiled=args.tiled_vae,
                LQ_video=LQ_tile,
                num_frames=F,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=topk_ratio,
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                color_fix=args.color_fix,
                unload_dit=args.unload_dit,
            )
        
        processed_tile_cpu = tensor2video(output_tile).to("cpu")
        
        mask_nchw = create_feather_mask(
            (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
            args.tile_overlap * args.scale,
        ).to("cpu")
        mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
        
        results.append({
            'coords': (x1, y1, x2, y2),
            'tile': processed_tile_cpu,
            'mask': mask_nhwc
        })
        
        del LQ_tile, output_tile, processed_tile_cpu, input_tile
        clean_vram()
    
    return results

def determine_optimal_batch_size(device: str, tile_coords: List[Tuple[int, int, int, int]], 
                                  frames: torch.Tensor, args) -> int:
    """Determine optimal batch size based on available GPU memory."""
    if not args.adaptive_batch_size or not device.startswith("cuda:"):
        return 1
    
    # 获取模型加载后的实际可用显存
    available_gb = get_available_memory_gb(device)
    used_gb, total_gb = get_gpu_memory_info(device)
    N = frames.shape[0]
    
    # 估算单个tile所需内存
    tile_size = args.tile_size
    dtype_size = 2 if args.precision in ["fp16", "bf16"] else 4
    tile_memory = estimate_tile_memory(tile_size, N, args.scale, dtype_size)
    
    # 对于大显存GPU（>=24GB），使用更激进的安全边界（只保留1GB）
    # 对于小显存GPU，保留2GB安全边界
    if total_gb >= 24:
        safe_memory = max(1.0, available_gb - 1.0)
        max_batch_limit = 16  # 32GB GPU可以支持更多并发
    else:
        safe_memory = max(2.0, available_gb - 2.0)
        max_batch_limit = 8
    
    # 计算可以同时处理的tile数量
    max_batch = max(1, int(safe_memory / tile_memory))
    optimal_batch = min(max_batch, max_batch_limit, len(tile_coords))
    
    if optimal_batch > 1:
        log(f"[Optimization] GPU: {device}, Total: {total_gb:.1f}GB, Used: {used_gb:.1f}GB, "
            f"Available: {available_gb:.2f}GB", "info")
        log(f"[Optimization] Estimated per-tile: {tile_memory:.2f}GB, "
            f"Safe memory: {safe_memory:.2f}GB, Using batch_size={optimal_batch}", "info")
    
    return optimal_batch

def run_inference(pipe, frames, device, dtype, args):
    """Run inference; 支持整图与 DiT 瓦片两种路径（与 nodes.py 对齐）。
    
    新增功能：
    1. 动态batch_size：根据显存情况同时处理多个tile
    """
    # 基本输入校验
    if frames is None or not hasattr(frames, 'shape') or frames.ndim != 4 or frames.shape[0] == 0:
        raise RuntimeError("[run_inference] Input frames is empty. Please check video decoding and path.")
    # 确保最少 21 帧（便于首尾填充）
    if frames.shape[0] < 21:
        add = 21 - frames.shape[0]
        last_frame = frames[-1:, :, :, :]
        padding_frames = last_frame.repeat(add, 1, 1, 1)
        frames = torch.cat([frames, padding_frames], dim=0)

    # 瓦片 DiT 路径：参考 nodes.py 的实现
    if args.tiled_dit:
        N, H, W, C = frames.shape
        num_aligned_frames = largest_8n1_leq(N + 4) - 4

        final_output_canvas = torch.zeros(
            (num_aligned_frames, H * args.scale, W * args.scale, C),
            dtype=torch.float32,
            device="cpu",
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)

        tile_coords = calculate_tile_coords(H, W, args.tile_size, args.tile_overlap)
        
        # 确定最优batch_size
        batch_size = determine_optimal_batch_size(device, tile_coords, frames, args)
        
        # 将tile_coords分成batch
        tile_batches = [tile_coords[i:i + batch_size] 
                       for i in range(0, len(tile_coords), batch_size)]
        
        total_tiles = len(tile_coords)
        processed = 0
        
        for batch_idx, tile_batch in enumerate(tqdm(tile_batches, desc="Processing Tile Batches")):
            # 处理当前batch的tiles
            results = process_tile_batch(pipe, frames, device, dtype, args, tile_batch, batch_idx)
            
            # 合并结果到canvas
            for result in results:
                x1, y1, x2, y2 = result['coords']
                processed_tile_cpu = result['tile']
                mask_nhwc = result['mask']
                
                out_x1, out_y1 = x1 * args.scale, y1 * args.scale
                tile_H_scaled = processed_tile_cpu.shape[1]
                tile_W_scaled = processed_tile_cpu.shape[2]
                out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
                
                final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
                
                processed += 1
                log(f"[FlashVSR] Processed tile {processed}/{total_tiles}: ({x1},{y1})-({x2},{y2})", "info")
            
            # 每次batch后清理显存
            clean_vram()
            
            # 动态调整batch_size（如果启用）
            if args.adaptive_batch_size and batch_idx > 0 and batch_idx % 5 == 0:
                new_batch_size = determine_optimal_batch_size(device, tile_coords[processed:], frames, args)
                if new_batch_size != batch_size:
                    batch_size = new_batch_size
                    log(f"[Optimization] Adjusted batch_size to {batch_size}", "info")

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
        return final_output

    # 整图路径
    LQ, th, tw, F = prepare_input_tensor(frames, device, scale=args.scale, dtype=dtype)
    if "long" not in args.mode:
        LQ = LQ.to(device)

    topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)

    with torch.no_grad():
        output = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            tiled=args.tiled_vae,
            progress_bar_cmd=tqdm,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=topk_ratio,
            kv_ratio=args.kv_ratio,
            local_range=args.local_range,
            color_fix=args.color_fix,
            unload_dit=args.unload_dit,
        )

    if isinstance(output, (tuple, list)):
        output = output[0]

    final_output = tensor2video(output).to("cpu")
    del output, LQ
    clean_vram()
    return final_output

# ==============================================================
#                            Main
# ==============================================================

def main(args):
    total_start = time.time()
    # 处理多GPU模式
    if args.multi_gpu:
        # 获取所有可用的CUDA设备
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for multi-GPU processing!")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            log(f"Warning: Only {num_gpus} GPU(s) available, falling back to single GPU mode", "warning")
            args.multi_gpu = False
        else:
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            log(f"[Multi-GPU] Using {num_gpus} GPUs: {devices}", "info")
            
            # 处理 attention_mode
            if args.attention_mode == "sparse_sage_attention":
                wan_video_dit.USE_BLOCK_ATTN = False
            else:
                wan_video_dit.USE_BLOCK_ATTN = True
            
            # 读取视频
            frames = read_video_to_tensor(args.input)
            
            # 使用多GPU处理
            output = run_inference_multi_gpu(frames, devices, args)
            
            # 保存视频
            output_dir = args.output if args.output else os.path.join("results", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
            save_video(output, output_dir)
            
            del frames, output
            clean_vram()
            
            total_time = time.time() - total_start
            log(f"Output saved to {output_dir}", "finish")
            log(f"[Total] Total elapsed time: {format_duration(total_time)}", "finish")
            return
    
    # 单GPU模式
    _device = args.device
    if _device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch, "mps") and torch.mps.is_available() else "cpu"
    
    if _device == "auto":
        raise RuntimeError("No devices found to run FlashVSR!")
    
    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
    
    if args.tiled_dit and (args.tile_overlap > args.tile_size / 2):
        raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
    
    # 处理 attention_mode
    if args.attention_mode == "sparse_sage_attention":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True
    
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.precision, torch.bfloat16)
    
    pipe = init_pipeline(args.mode, _device, dtype, args.model_dir)
    frames = read_video_to_tensor(args.input)
    
    output = run_inference(pipe, frames, _device, dtype, args)
    
    # 保存视频
    output_dir = args.output if args.output else os.path.join("results", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
    save_video(output, output_dir)
    
    del pipe, frames, output
    clean_vram()
    
    total_time = time.time() - total_start
    log(f"Output saved to {output_dir}", "finish")
    log(f"[Total] Total elapsed time: {format_duration(total_time)}", "finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashVSR Standalone Inference with Optimizations")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--model_dir", type=str, default="/app/FlashVSR/examples/WanVSR/FlashVSR", help="Model directory")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "full", "tiny-long"], help="Model mode")
    parser.add_argument("--device", type=str, default="cuda:0", choices=get_device_list(), help="Device (ignored if --multi_gpu is used)")
    
    # 从 nodes.py 添加所有参数
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor")
    parser.add_argument("--color_fix", type=bool, default=True, help="Use color fix")
    parser.add_argument("--tiled_vae", type=bool, default=True, help="Use tiled VAE")
    parser.add_argument("--tiled_dit", type=bool, default=False, help="Use tiled DiT")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--tile_overlap", type=int, default=24, help="Tile overlap")
    parser.add_argument("--unload_dit", type=bool, default=False, help="Unload DiT before decoding")
    parser.add_argument("--sparse_ratio", type=float, default=2.0, help="Sparse ratio")
    parser.add_argument("--kv_ratio", type=float, default=3.0, help="KV ratio")
    parser.add_argument("--local_range", type=int, default=11, help="Local range")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--attention_mode", type=str, default="sparse_sage_attention", choices=["sparse_sage_attention", "block_sparse_attention"], help="Attention mode")
    
    # 新增优化参数
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU parallel processing (splits video by frames)")
    parser.add_argument("--adaptive_batch_size", action="store_true", help="Enable adaptive batch size for tiles (dynamically adjust based on GPU memory)")
    
    args = parser.parse_args()
    
    main(args)
