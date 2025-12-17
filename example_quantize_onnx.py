"""
Simple example: How to quantize an exported ONNX model

This script demonstrates the basic usage of quantizing ONNX models
for deployment on Ryzen AI NPU or other platforms.
"""

import os
import numpy as np
from pathlib import Path

# Configuration
ONNX_MODEL_PATH = "./models/qwen3-coder-30b-onnx"
QUANTIZED_OUTPUT_PATH = "./models/qwen3-coder-30b-onnx-int8"

def create_calibration_data(tokenizer, num_samples=100, seq_length=128):
    """
    Create calibration data for quantization.
    This should be representative of your actual input data.
    
    Args:
        tokenizer: Hugging Face tokenizer
        num_samples: Number of calibration samples
        seq_length: Sequence length for each sample
    """
    calibration_data = []
    
    # Example prompts (replace with your actual use cases)
    example_prompts = [
        "Write a Python function to",
        "def calculate",
        "import numpy as np",
        "class MyClass",
        "if __name__ == '__main__':",
    ]
    
    for i in range(num_samples):
        # Use example prompts or generate random tokens
        prompt = example_prompts[i % len(example_prompts)]
        
        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=seq_length,
            truncation=True
        )
        
        # Convert to the format expected by quantization
        calibration_data.append({
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs.get("attention_mask", np.ones_like(inputs["input_ids"])).astype(np.int64),
        })
    
    return calibration_data


def quantize_onnx_simple():
    """
    Simple example of quantizing an ONNX model using ONNX Runtime.
    """
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
    except ImportError:
        print("❌ Error: onnxruntime not installed")
        print("   Install with: pip install onnxruntime")
        return False
    
    # Find the ONNX model file
    if os.path.isdir(ONNX_MODEL_PATH):
        onnx_files = list(Path(ONNX_MODEL_PATH).glob("*.onnx"))
        if not onnx_files:
            print(f"❌ No .onnx files found in {ONNX_MODEL_PATH}")
            print("   Make sure you've exported the model to ONNX first")
            return False
        model_file = str(max(onnx_files, key=lambda p: p.stat().st_size))
    else:
        model_file = ONNX_MODEL_PATH
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
    
    print(f"Found ONNX model: {model_file}")
    
    # Prepare output path
    os.makedirs(QUANTIZED_OUTPUT_PATH, exist_ok=True)
    if os.path.isdir(QUANTIZED_OUTPUT_PATH):
        output_file = os.path.join(QUANTIZED_OUTPUT_PATH, os.path.basename(model_file))
    else:
        output_file = QUANTIZED_OUTPUT_PATH
    
    # Create calibration data
    # Option 1: Load tokenizer and create proper calibration data
    try:
        from transformers import AutoTokenizer
        tokenizer_path = "./models/Qwen3-Coder-30B-A3B-Instruct"
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            print("✓ Loaded tokenizer for calibration data")
            calibration_data = create_calibration_data(tokenizer)
        else:
            print("⚠ Tokenizer not found, using dummy calibration data")
            # Option 2: Use dummy data (less accurate but works)
            calibration_data = [
                {"input_ids": np.random.randint(0, 1000, size=(1, 128), dtype=np.int64)}
                for _ in range(100)
            ]
    except Exception as e:
        print(f"⚠ Could not load tokenizer: {e}")
        print("   Using dummy calibration data")
        calibration_data = [
            {"input_ids": np.random.randint(0, 1000, size=(1, 128), dtype=np.int64)}
            for _ in range(100)
        ]
    
    # Create calibration data reader
    class DataReader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.iterator = iter(self.data)
        
        def get_next(self):
            return next(self.iterator, None)
        
        def rewind(self):
            self.iterator = iter(self.data)
    
    data_reader = DataReader(calibration_data)
    
    print(f"\nQuantizing model...")
    print(f"  Input: {model_file}")
    print(f"  Output: {output_file}")
    print(f"  Calibration samples: {len(calibration_data)}")
    
    try:
        quantize_static(
            model_input=model_file,
            model_output=output_file,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format
            per_channel=True,  # Per-channel quantization for weights
            reduce_range=True,  # Reduce quantization range for compatibility
            weight_type=QuantType.QInt8,  # INT8 weights
            activation_type=QuantType.QUInt8,  # UINT8 activations
            optimize_model=True,  # Optimize after quantization
        )
        
        print(f"\n✓ Quantization completed!")
        print(f"  Quantized model saved to: {output_file}")
        
        # Show size comparison
        if os.path.exists(output_file):
            original_size = os.path.getsize(model_file) / (1024**3)
            quantized_size = os.path.getsize(output_file) / (1024**3)
            reduction = (1 - quantized_size / original_size) * 100
            print(f"\nSize comparison:")
            print(f"  Original: {original_size:.2f} GB")
            print(f"  Quantized: {quantized_size:.2f} GB")
            print(f"  Reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("ONNX Model Quantization Example")
    print("=" * 70)
    print(f"\nInput model: {ONNX_MODEL_PATH}")
    print(f"Output path: {QUANTIZED_OUTPUT_PATH}")
    print("\nThis example uses ONNX Runtime quantization (INT8/QDQ format)")
    print("For AMD Quark (INT4/BF16 for NPU), use quantize_onnx_model.py with --method quark\n")
    
    success = quantize_onnx_simple()
    
    if success:
        print("\n" + "=" * 70)
        print("Next steps:")
        print("=" * 70)
        print("1. Test the quantized model to verify accuracy")
        print("2. For Ryzen AI NPU: Use VitisAIExecutionProvider in ONNX Runtime")
        print("3. Compare inference speed and memory usage")
        print("\nExample usage with ONNX Runtime:")
        print("  from onnxruntime import InferenceSession")
        print("  session = InferenceSession('quantized_model.onnx')")
    else:
        print("\nQuantization failed. Check the error messages above.")

