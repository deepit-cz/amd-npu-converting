# ONNX Model Quantization Guide

This guide explains how to quantize exported ONNX models for deployment, especially for Ryzen AI NPU.

## Overview

Quantization reduces model size and improves inference speed by converting floating-point weights and activations to lower precision formats (INT8, INT4, BF16). This is especially important for NPU deployment where memory and speed are critical.

## Quick Start

### Method 1: Simple Example Script

```bash
python example_quantize_onnx.py
```

This uses the default ONNX Runtime quantization (INT8) and works on any platform.

### Method 2: Full-Featured Script

```bash
# ONNX Runtime quantization (INT8)
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int8 \
    --method onnxruntime

# AMD Quark quantization (INT4 for NPU)
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int4 \
    --method quark \
    --quant-type int4
```

## Quantization Methods

### 1. ONNX Runtime Quantization (Recommended for General Use)

**Best for:** General deployment, CPU/GPU inference, cross-platform compatibility

**Features:**
- INT8 quantization (weights and activations)
- QDQ (Quantize-Dequantize) or QOperator format
- Per-channel quantization for weights
- Works on any platform with ONNX Runtime

**Installation:**
```bash
pip install onnxruntime
# For GPU support:
pip install onnxruntime-gpu
```

**Usage:**
```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

quantize_static(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)
```

**Pros:**
- ✅ Widely supported
- ✅ Good accuracy retention
- ✅ Easy to use
- ✅ Works everywhere

**Cons:**
- ❌ INT8 only (not INT4)
- ❌ May not be optimized for specific NPU hardware

### 2. AMD Quark Quantization (Recommended for Ryzen AI NPU)

**Best for:** Ryzen AI NPU deployment, maximum compression (INT4), BF16 support

**Features:**
- INT4, INT8, or BF16 quantization
- Optimized for Ryzen AI NPU
- Better compression ratios
- NPU-specific optimizations

**Installation:**
Quark is included with AMD Ryzen AI Software. Download from:
- [AMD Ryzen AI Documentation](https://ryzenai.docs.amd.com)
- Install the complete Ryzen AI Software suite

**Usage:**
```python
from quark.onnx import quantize

quantize(
    model_path="model.onnx",
    output_path="model_quantized.onnx",
    quantization_type="int4",  # or "int8", "bf16"
)
```

**Pros:**
- ✅ INT4 quantization (smallest models)
- ✅ BF16 support for transformers
- ✅ Optimized for Ryzen AI NPU
- ✅ Best performance on NPU hardware

**Cons:**
- ❌ Requires AMD Ryzen AI Software
- ❌ Platform-specific (AMD NPU)

**When to use:**
- Deploying to Ryzen AI NPU
- Need maximum compression (INT4)
- Want NPU-optimized performance

### 3. ONNX Neural Compressor

**Best for:** Advanced quantization options, research, alternative tool

**Features:**
- Multiple quantization approaches
- Advanced calibration methods
- Model optimization
- Cross-platform support

**Installation:**
```bash
pip install neural-compressor
```

**Usage:**
```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

config = PostTrainingQuantConfig(approach="static")
quantized_model = quantization.fit(
    model="model.onnx",
    conf=config,
    calib_dataloader=calibration_dataset,
)
```

**Pros:**
- ✅ Advanced features
- ✅ Multiple quantization strategies
- ✅ Good for research

**Cons:**
- ❌ More complex API
- ❌ Less commonly used

## Calibration Data

**Important:** For best accuracy, provide representative calibration data that matches your actual use case.

### Creating Calibration Data

```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("model_path", trust_remote_code=True)

calibration_data = []
for prompt in your_prompts:
    inputs = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    calibration_data.append({
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs.get("attention_mask").astype(np.int64),
    })
```

### Saving/Loading Calibration Data

```python
# Save
import pickle
with open("calibration_data.pkl", "wb") as f:
    pickle.dump(calibration_data, f)

# Or as numpy
np.savez("calibration_data.npz", **{k: v for d in calibration_data for k, v in d.items()})

# Load
with open("calibration_data.pkl", "rb") as f:
    calibration_data = pickle.load(f)
```

## Quantization Types Comparison

| Type | Size Reduction | Accuracy | Speed | Use Case |
|------|---------------|----------|-------|----------|
| **INT4** | ~75% | Good | Fastest | NPU deployment, maximum compression |
| **INT8** | ~50% | Very Good | Fast | General deployment, good balance |
| **BF16** | ~50% | Excellent | Fast | Transformer models, NPU support |

## Step-by-Step Workflow

### 1. Export Model to ONNX

First, export your model to ONNX format:

```bash
python load_model_from_local_folder.py
# Set EXPORT_TO_ONNX = True
```

This creates ONNX files in `./models/qwen3-coder-30b-onnx/`

### 2. Prepare Calibration Data

Create representative calibration data from your actual use cases:

```python
# See example_quantize_onnx.py for details
calibration_data = create_calibration_data(tokenizer, num_samples=100)
```

### 3. Quantize the Model

Choose your quantization method:

**For general use (INT8):**
```bash
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int8 \
    --method onnxruntime
```

**For Ryzen AI NPU (INT4):**
```bash
python quantize_onnx_model.py \
    --input ./models/qwen3-coder-30b-onnx \
    --output ./models/qwen3-coder-30b-onnx-int4 \
    --method quark \
    --quant-type int4
```

### 4. Validate Quantized Model

Test the quantized model to ensure accuracy:

```python
from onnxruntime import InferenceSession
import numpy as np

# Load quantized model
session = InferenceSession("quantized_model.onnx")

# Test inference
inputs = {"input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64)}
outputs = session.run(None, inputs)
```

### 5. Deploy on NPU (Ryzen AI)

For Ryzen AI NPU deployment:

```python
from onnxruntime import InferenceSession
from onnxruntime.vitisai import VitisAIExecutionProvider

session = InferenceSession(
    "quantized_model.onnx",
    providers=['VitisAIExecutionProvider']
)
```

## Troubleshooting

### "No calibration data provided"

**Solution:** Provide representative calibration data. The script will use dummy data if none is provided, but accuracy may suffer.

```bash
python quantize_onnx_model.py \
    --input model.onnx \
    --output model_quantized.onnx \
    --calibration-data calibration_data.pkl
```

### "AMD Quark not installed"

**Solution:** Install AMD Ryzen AI Software from [AMD Ryzen AI Documentation](https://ryzenai.docs.amd.com). Quark is included in the software suite.

### "Precision errors" or "Type mismatch"

**Solution:** This often happens when exporting quantized models. Re-quantize the ONNX model:

```bash
# Export non-quantized model first
# Then quantize the ONNX model
python quantize_onnx_model.py --input model.onnx --output model_quantized.onnx
```

### "Model too large for quantization"

**Solution:** 
- Use INT4 quantization for maximum compression
- Split the model if possible
- Ensure sufficient disk space (quantization creates temporary files)

### Accuracy degradation

**Solutions:**
1. Use more calibration data (100-1000 samples)
2. Ensure calibration data is representative
3. Try different quantization types (INT8 vs INT4)
4. Use BF16 if supported (better accuracy)

## Best Practices

1. **Always validate:** Test quantized models with your actual use cases
2. **Use representative data:** Calibration data should match real inputs
3. **Start with INT8:** If unsure, start with INT8 quantization (better accuracy)
4. **For NPU:** Use AMD Quark with INT4 for best NPU performance
5. **Compare results:** Always compare quantized vs original model accuracy
6. **Monitor size:** Check file size reduction to ensure quantization worked

## File Size Expectations

For a 30B parameter model:
- **Original ONNX (FP32/BF16):** ~60 GB
- **INT8 quantized:** ~30 GB (50% reduction)
- **INT4 quantized:** ~15 GB (75% reduction)

## Additional Resources

- [ONNX Runtime Quantization Docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [AMD Quark Documentation](https://quark.docs.amd.com)
- [AMD Ryzen AI Documentation](https://ryzenai.docs.amd.com)
- [ONNX Neural Compressor](https://github.com/onnx/neural-compressor)

## Summary

- **For general use:** ONNX Runtime quantization (INT8) - works everywhere
- **For Ryzen AI NPU:** AMD Quark quantization (INT4/BF16) - optimized for NPU
- **For research:** ONNX Neural Compressor - advanced options

Always provide representative calibration data for best accuracy!

