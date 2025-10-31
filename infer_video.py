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
    """Read video and convert to (N, H, W, C) format for prepare_input_tensor."""
    vr = torchvision.io.read_video(video_path, pts_unit='sec')[0]
    vr = vr.permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W)
    vr = vr.permute(0, 2, 3, 1)  # (N, H, W, C)
    return vr

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

def run_inference(pipe, frames, device, dtype, args):
    """Run inference; 支持整图与 DiT 瓦片两种路径（与 nodes.py 对齐）。"""
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

        for i, (x1, y1, x2, y2) in enumerate(tqdm(tile_coords, desc="Processing Tiles")):
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: ({x1},{y1})-({x2},{y2})", "info")
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

            # 混合权重（羽化）
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                args.tile_overlap * args.scale,
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)

            out_x1, out_y1 = x1 * args.scale, y1 * args.scale
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled

            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc

            del LQ_tile, output_tile, processed_tile_cpu, input_tile
            clean_vram()

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
    
    log(f"Output saved to {output_dir}", "finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashVSR Standalone Inference")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--model_dir", type=str, default="/app/FlashVSR/examples/WanVSR/FlashVSR", help="Model directory")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "full", "tiny-long"], help="Model mode")
    parser.add_argument("--device", type=str, default="cuda:0", choices=get_device_list(), help="Device")
    
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
    
    args = parser.parse_args()
    
    main(args)
