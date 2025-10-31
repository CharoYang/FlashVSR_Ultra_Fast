# FlashVSR 优化指南

本文档说明了 `infer_video.py` 中新增的两个优化功能的用法和原理。

## 优化功能

### 1. 多GPU并行处理 (`--multi_gpu`)

**功能说明：**
将视频按帧切分成 N 部分（N 为 GPU 数量），每个 GPU 处理一部分，最后合并结果。这可以显著加速视频处理。

**使用方法：**
```bash
python infer_video.py --input video.mp4 --multi_gpu --tiled_dit
```

**工作原理：**
- 自动检测所有可用的 CUDA GPU
- 将视频帧分割成多个 segments（每个 GPU 处理一个 segment）
- segments 之间有少量重叠（默认 10 帧），确保边界平滑
- 每个 GPU 独立处理自己的 segment
- 处理完成后自动合并所有 segments

**注意事项：**
- 需要至少 2 个 GPU 才能启用
- 每个 GPU 都会加载完整的模型（显存需求不变）
- 适用于长时间视频（帧数多），短视频可能不会显著加速

**示例：**
```bash
# 使用所有可用的 GPU 处理视频
python infer_video.py \
    --input long_video.mp4 \
    --output output.mp4 \
    --multi_gpu \
    --tiled_dit \
    --tile_size 256 \
    --tile_overlap 24
```

---

### 2. 动态 Batch Size 调整 (`--adaptive_batch_size`)

**功能说明：**
根据 GPU 显存状况动态调整同时处理的 tile 数量。当显存充足时，可以同时处理多个 tile 以提高效率。

**使用方法：**
```bash
python infer_video.py --input video.mp4 --tiled_dit --adaptive_batch_size
```

**工作原理：**
- 在开始处理 tiles 前，检测 GPU 可用显存
- 根据每个 tile 的预估显存需求，计算最优 batch_size（1-8）
- 每处理 5 个 batch 后，重新检测显存并调整 batch_size
- 如果显存不足，batch_size 会自动降低；显存充足时可以提高

**显存监控：**
- 使用 `torch.cuda.memory_reserved()` 获取 GPU 显存使用情况
- 保留 2GB 显存作为安全边界，避免 OOM
- 限制最大 batch_size 为 8，避免显存溢出

**注意事项：**
- 只在启用 `--tiled_dit` 时有效
- 需要 CUDA GPU（不支持 MPS）
- 实际加速效果取决于显存大小和 tile 数量

**示例：**
```bash
# 启用动态 batch_size，自动调整 tile 批处理大小
python infer_video.py \
    --input video.mp4 \
    --output output.mp4 \
    --tiled_dit \
    --adaptive_batch_size \
    --tile_size 256 \
    --tile_overlap 24 \
    --device cuda:0
```

---

## 组合使用

两个优化功能可以同时使用：

```bash
# 多GPU + 动态batch_size（最佳性能）
python infer_video.py \
    --input long_video.mp4 \
    --output output.mp4 \
    --multi_gpu \
    --tiled_dit \
    --adaptive_batch_size \
    --tile_size 256 \
    --tile_overlap 24 \
    --model_dir /path/to/models
```

**优化效果：**
- **多GPU并行**：加速比 ≈ GPU 数量（理想情况）
- **动态batch_size**：在小显存场景下可提升 2-4x 效率
- **组合使用**：可以获得接近线性的多GPU加速效果

---

## 性能建议

1. **短视频（< 100 帧）**：
   - 使用 `--adaptive_batch_size` 即可
   - 多GPU可能不会带来明显加速（模型加载开销）

2. **中等视频（100-500 帧）**：
   - 如果有多GPU，使用 `--multi_gpu`
   - 启用 `--adaptive_batch_size` 进一步提升效率

3. **长视频（> 500 帧）**：
   - 强烈推荐同时使用两个优化
   - 多GPU并行效果最明显

---

## 故障排除

### 多GPU 问题

**问题：** 进程卡死或出错
- 检查所有 GPU 是否可用：`nvidia-smi`
- 确保每个 GPU 有足够显存加载模型
- 检查是否有其他进程占用 GPU

**问题：** 合并后的视频有接缝
- 增加 `split_video_by_frames` 中的 `overlap` 参数（默认为 10）
- 检查视频帧数是否足够多

### 动态 Batch Size 问题

**问题：** batch_size 始终为 1
- 检查是否启用了 `--adaptive_batch_size`
- 检查是否有足够的显存（至少需要 4GB 空闲显存）
- 查看日志中的显存信息

**问题：** 显存溢出（OOM）
- 降低 `--tile_size`
- 增加 `--tile_overlap`（可能降低处理速度）
- 使用 `--unload_dit` 选项

---

## 技术细节

### 显存估算公式

```
tile_memory ≈ (tile_size² × num_frames × 3 × dtype_size) × 13
```

其中：
- `dtype_size = 2` (fp16/bf16) 或 `4` (fp32)
- 系数 13 包括输入、输出和中间激活层（保守估计）

### 多GPU分割策略

视频按帧数均匀分割：
- Segment 大小 = `总帧数 / GPU数量`
- 每个 segment 前后有 10 帧重叠（可调整）
- 最后一个 segment 可能稍短

### 合并策略

- 移除 segments 之间的重叠部分
- 使用线性混合处理边界（如果有需要）
- 确保最终视频帧数与原始视频一致

---

## 更新日志

- **v1.0** (当前版本)
  - 添加多GPU并行处理
  - 添加动态batch_size调整
  - 添加显存监控功能

