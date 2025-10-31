# ComfyUI-FlashVSR_Ultra_Fast
Running FlashVSR on lower VRAM without any artifacts.   
**[[📃中文版本](./README_zh.md)]**

## Changelog
#### 2025-10-31
- **New:** Standalone `infer_video.py` script for video processing without ComfyUI
- **New:** Multi-GPU parallel processing (`--multi_gpu`) - automatically splits video by frames across GPUs
- **New:** Adaptive tile batch size (`--adaptive_batch_size`) - dynamically adjusts tile concurrency based on GPU memory
- **New:** GPU memory monitoring and optimization for 24-32GB GPUs
- **New:** Total elapsed time tracking for performance monitoring
- **Improved:** Video reading with OpenCV fallback for better codec compatibility
- **Fixed:** Empty frame handling and negative dimension errors

#### 2025-10-24
- Added long video pipeline that significantly reduces VRAM usage when upscaling long videos.

#### 2025-10-22
- Replaced `Block-Sparse-Attention` with `Sparse_Sage`, removing the need to compile any custom kernels.  
- Added support for running on RTX 50 series GPUs.

#### 2025-10-21
- Initial this project, introducing features such as `tile_dit` to significantly reducing VRAM usage.

## Preview
![](./img/preview.jpg)

## Usage

### Standalone Inference (New!)

Use `infer_video.py` for standalone video processing without ComfyUI:

```bash
python infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit true \
  --tile_size 256 \
  --tile_overlap 64 \
  --multi_gpu \
  --adaptive_batch_size \
  --model_dir /path/to/FlashVSR
```

**Key Parameters:**
- **`--multi_gpu`**: Enable multi-GPU parallel processing (splits video by frames across GPUs)
- **`--adaptive_batch_size`**: Enable adaptive tile batch size based on available GPU memory
- **`--model_dir`**: Path to FlashVSR model directory (default: `/app/FlashVSR/examples/WanVSR/FlashVSR`)

For detailed optimization guide, see [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md).

### ComfyUI Nodes

- **mode:**  
`tiny` -> faster (default); `full` -> higher quality  
- **scale:**  
`4` is always better, unless you are low on VRAM then use `2`    
- **color_fix:**  
Use wavelet transform to correct the color of output video.  
- **tiled_vae:**  
Set to True for lower VRAM consumption during decoding at the cost of speed.  
- **tiled_dit:**  
Significantly reduces VRAM usage at the cost of speed.
- **tile\_size, tile\_overlap**:  
How to split the input video.  
- **unload_dit:**  
Unload DiT before decoding to reduce VRAM peak at the cost of speed.  

## Installation

#### nodes: 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```
> 📢: For Turing or older GPU, please install `triton<3.3.0`:  

> ```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### models:

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in the `ComfyUI/models`

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Performance Optimization

### Multi-GPU Processing
For systems with 2+ GPUs, enable `--multi_gpu` to achieve near-linear speedup:
- Automatically splits video frames across available GPUs
- Each GPU processes a video segment independently
- Results are merged seamlessly with overlap handling

### Adaptive Batch Size
Enable `--adaptive_batch_size` to maximize GPU utilization:
- Dynamically adjusts tile batch size based on available VRAM
- For 32GB GPUs, can process 6-16 tiles concurrently
- Automatically rebalances during processing if memory changes

**Expected Performance:**
- **2 GPUs + Adaptive Batch**: 3-5x speedup compared to single GPU
- **VRAM Usage**: 20-25GB peak on 32GB GPUs (vs 13GB without optimizations)

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
