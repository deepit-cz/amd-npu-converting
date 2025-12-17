# Multi-Machine Quantization Workflow

## Your Setup
- **Linux Machine**: 512GB RAM, 2x Intel Xeon CPUs → Use for INT8 quantization
- **Ryzen 9950X**: 96GB RAM, 12GB VRAM → **Cannot use Quark (no NPU)**
- **Laptop with NPU**: 32GB RAM → Use for INT4 quantization (if INT8 fits)

## Problem
Your original workflow won't work because:
- **Ryzen 9950X has NO NPU** → AMD Quark requires NPU
- INT4 model (~44GB) won't fit in 32GB RAM laptop

## Recommended Workflow

### Option 1: Two-Step Process (Recommended)

#### Step 1: INT8 on Linux Machine (512GB RAM)
```bash
# On Linux machine with 512GB RAM
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int8 \
    --method onnxruntime
```

**Expected result:**
- INT8 model: ~88GB total (should fit in 96GB RAM on 9950X)
- Time: Several hours

#### Step 2: INT4 on Laptop with NPU (32GB RAM)
**Problem**: INT8 model (~88GB) won't fit in 32GB RAM laptop!

**Solution**: Use the **original FP32 model** on laptop (if you can transfer it)

```bash
# On laptop with NPU (32GB RAM)
# Transfer original FP32 model from Linux machine
# Then quantize directly to INT4:
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int4 \
    --method quark \
    --quant-type int4
```

**Expected result:**
- INT4 model: ~44GB total
- **Still won't fit in 32GB RAM for loading!**

### Option 2: Use INT8 on Laptop (Fits in 32GB)

**Best solution**: Use INT8 model on laptop (fits in 32GB RAM)

```bash
# On laptop with NPU
# Use INT8 model (transferred from Linux)
# INT8 model: ~88GB on disk, but can be loaded with memory mapping
```

**Note**: ONNX Runtime can use memory mapping to load large models without loading everything into RAM.

### Option 3: Quantize Directly to INT4 on Linux (If Quark Works)

If you can install Quark on Linux machine (unlikely without NPU), or if there's a CPU version:

```bash
# On Linux machine (512GB RAM)
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int4 \
    --method quark \
    --quant-type int4
```

## Size Breakdown

| Model Type | ONNX File | onnx_data | Total | Fits in 32GB? |
|------------|-----------|----------|-------|---------------|
| **Original (FP32)** | 61GB | 116GB | 177GB | ❌ No |
| **INT8** | ~30GB | ~58GB | ~88GB | ⚠️ With memory mapping |
| **INT4** | ~15GB | ~29GB | ~44GB | ❌ No (too large) |

## Memory Mapping Solution

Even if model is larger than RAM, you can use memory mapping:

```python
from onnxruntime import InferenceSession

# ONNX Runtime uses memory mapping automatically
# Model doesn't need to fit entirely in RAM
session = InferenceSession(
    "model_int4.onnx",
    providers=['VitisAIExecutionProvider', 'CPUExecutionProvider']
)
```

## Recommended Final Workflow

1. **Linux Machine (512GB RAM)**: Quantize to INT8
   ```bash
   python quantize_onnx_model.py --input ./models/qwen3-coder-30b-onnx \
       --output ./models/qwen3-coder-30b-onnx-int8 --method onnxruntime
   ```

2. **Transfer INT8 model to laptop** (~88GB)

3. **Laptop with NPU (32GB RAM)**: 
   - Use INT8 model with memory mapping
   - OR: If you can get original FP32 model, quantize to INT4
   - INT4 model (~44GB) can be used with memory mapping

4. **For inference**: Use memory mapping - model doesn't need to fit entirely in RAM

## Alternative: Use INT8 on Laptop

**Simplest solution**: Just use INT8 model on laptop:
- INT8: ~88GB (fits with memory mapping)
- Good accuracy
- Works with NPU (Vitis AI EP supports INT8)
- No need for INT4 conversion

## Summary

- ✅ **INT8 on Linux**: Works perfectly
- ❌ **INT4 on 9950X**: Won't work (no NPU for Quark)
- ⚠️ **INT4 on Laptop**: Model too large for RAM, but memory mapping can help
- ✅ **INT8 on Laptop**: Best option - fits with memory mapping

**Recommendation**: Use INT8 model on laptop with memory mapping. It's simpler and works well with NPU.

