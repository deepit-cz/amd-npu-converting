"""
Script to quantize exported ONNX models for Ryzen AI NPU and other targets.

This script provides multiple quantization methods:
1. AMD Quark quantization (INT4/BF16) - Optimized for Ryzen AI NPU (RECOMMENDED for INT4)
2. ONNX Runtime quantization (INT8/QDQ) - Works on any platform (INT8 only)
3. ONNX Neural Compressor - Alternative quantization tool

Default: INT4 quantization using AMD Quark (best for NPU performance)

Usage:
    python quantize_onnx_model.py --input ./models/qwen3-coder-30b-onnx --output ./models/qwen3-coder-30b-onnx-int4
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

# ============================================================================
# Method 1: ONNX Runtime Quantization (INT8/QDQ only - INT4 not supported)
# ============================================================================

def quantize_with_onnxruntime(
    model_path: str,
    output_path: str,
    calibration_data: Optional[List[Dict[str, np.ndarray]]] = None,
    quant_format: str = "QDQ",  # QDQ or QOperator
    per_channel: bool = True,
    weight_type: str = "QInt8",  # Note: ONNX Runtime only supports INT8, not INT4
    activation_type: str = "QUInt8"  # Note: ONNX Runtime only supports INT8, not INT4
):
    """
    Quantize ONNX model using ONNX Runtime's quantization tools.
    
    NOTE: ONNX Runtime only supports INT8 quantization, not INT4.
    For INT4 quantization, use AMD Quark method instead.
    
    Args:
        model_path: Path to input ONNX model file or directory
        output_path: Path to save quantized model
        calibration_data: List of calibration data dictionaries
        quant_format: "QDQ" (Quantize-Dequantize) or "QOperator"
        per_channel: Enable per-channel quantization for weights
        weight_type: "QInt8" or "QUInt8" (INT8 only - INT4 not supported)
        activation_type: "QUInt8" or "QInt8" (INT8 only - INT4 not supported)
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
            QuantFormat
        )
    except ImportError:
        print("❌ Error: onnxruntime not installed or quantization module not available")
        print("   Install with: pip install onnxruntime")
        print("   For GPU support: pip install onnxruntime-gpu")
        return False
    
    print("=" * 70)
    print("Quantizing with ONNX Runtime (INT8 only)")
    print("=" * 70)
    print("⚠ Note: ONNX Runtime only supports INT8 quantization, not INT4.")
    print("   For INT4 quantization, use AMD Quark method instead.")
    print("=" * 70)
    
    # Find ONNX model file(s)
    if os.path.isdir(model_path):
        onnx_files = list(Path(model_path).glob("*.onnx"))
        if not onnx_files:
            print(f"❌ No .onnx files found in {model_path}")
            return False
        # Use the first/largest ONNX file (usually the main model)
        model_file = str(max(onnx_files, key=lambda p: p.stat().st_size))
        print(f"Found ONNX model: {model_file}")
    else:
        model_file = model_path
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
    
    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(output_path):
        quantized_model_path = os.path.join(output_path, os.path.basename(model_file))
    else:
        quantized_model_path = output_path
    
    # Create calibration data reader if provided
    class DataReader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data if data else []
            self.iterator = iter(self.data)
        
        def get_next(self):
            return next(self.iterator, None)
        
        def rewind(self):
            self.iterator = iter(self.data)
    
    # If no calibration data provided, create dummy data
    if calibration_data is None:
        print("⚠ Warning: No calibration data provided. Using dummy data.")
        print("   For better accuracy, provide representative calibration data.")
        # Create minimal dummy data (you should replace this with real data)
        try:
            import onnx
            model = onnx.load(model_file)
            # Get input shape from model
            input_shape = None
            for input_tensor in model.graph.input:
                if input_tensor.name == "input_ids" or len(model.graph.input) == 1:
                    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                    input_shape = [1 if d == 0 else d for d in shape]  # Replace dynamic dims with 1
                    break
            
            if input_shape:
                dummy_data = [{"input_ids": np.random.randint(0, 1000, size=input_shape, dtype=np.int64)}]
                calibration_data = dummy_data
            else:
                # Fallback: assume common shape
                calibration_data = [{"input_ids": np.random.randint(0, 1000, size=(1, 128), dtype=np.int64)}]
        except Exception as e:
            print(f"⚠ Could not infer input shape: {e}")
            calibration_data = [{"input_ids": np.random.randint(0, 1000, size=(1, 128), dtype=np.int64)}]
    
    data_reader = DataReader(calibration_data)
    
    # Map string types to QuantType
    type_map = {
        "QInt8": QuantType.QInt8,
        "QUInt8": QuantType.QUInt8,
        "QInt16": QuantType.QInt16,
        "QUInt16": QuantType.QUInt16,
    }
    
    format_map = {
        "QDQ": QuantFormat.QDQ,
        "QOperator": QuantFormat.QOperator,
    }
    
    try:
        print(f"\nQuantizing model...")
        print(f"  Input: {model_file}")
        print(f"  Output: {quantized_model_path}")
        print(f"  Format: {quant_format}")
        print(f"  Weight type: {weight_type}")
        print(f"  Activation type: {activation_type}")
        print(f"  Per-channel: {per_channel}")
        
        start_time = time.time()
        
        quantize_static(
            model_input=model_file,
            model_output=quantized_model_path,
            calibration_data_reader=data_reader,
            quant_format=format_map.get(quant_format, QuantFormat.QDQ),
            per_channel=per_channel,
            reduce_range=True,  # Reduce range for compatibility
            weight_type=type_map.get(weight_type, QuantType.QInt8),
            activation_type=type_map.get(activation_type, QuantType.QUInt8),
            # Note: optimize_model parameter not supported in this ONNX Runtime version
            # Model optimization can be done separately if needed
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Quantization completed in {elapsed:.2f} seconds")
        
        # Check file size
        if os.path.exists(quantized_model_path):
            size_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)
            original_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            reduction = (1 - size_mb / original_size_mb) * 100
            print(f"  Original size: {original_size_mb:.2f} MB")
            print(f"  Quantized size: {size_mb:.2f} MB")
            print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Method 1: AMD Quark Quantization (for Ryzen AI NPU) - RECOMMENDED for INT4
# ============================================================================

def check_if_model_is_quantized(model_path: str) -> tuple:
    """
    Check if an ONNX model is already quantized.
    
    Returns:
        (is_quantized, quantization_type)
        quantization_type: "int8", "int4", "unknown", or "fp32"
    """
    try:
        import onnx
        model = onnx.load(model_path)
        
        # Check for QDQ nodes (Quantize-Dequantize) - indicates INT8 quantization
        has_qdq = any(
            node.op_type in ["QuantizeLinear", "DequantizeLinear"] 
            for node in model.graph.node
        )
        
        # Check for INT8/INT4 weight types
        has_int8_weights = any(
            tensor.data_type in [3, 4]  # INT32, INT8
            for tensor in model.graph.initializer
        )
        
        if has_qdq or has_int8_weights:
            # Try to determine if INT4 or INT8
            # INT4 models often have specific patterns
            # For now, assume INT8 if QDQ present
            return True, "int8" if has_qdq else "unknown"
        
        return False, "fp32"
    except Exception:
        return False, "unknown"


def quantize_with_quark(
    model_path: str,
    output_path: str,
    quantization_type: str = "int4",  # int4 (default), int8, bf16
    calibration_data: Optional[List[Dict[str, np.ndarray]]] = None
):
    """
    Quantize ONNX model using AMD Quark Model Optimization Library.
    Optimized for Ryzen AI NPU deployment.
    
    NOTE: If input model is already quantized (INT8), Quark may or may not support
    re-quantization to INT4. Best practice: Use original FP32/FP16 model.
    
    Args:
        model_path: Path to input ONNX model file or directory
        output_path: Path to save quantized model
        quantization_type: "int4", "int8", or "bf16"
        calibration_data: Optional calibration data
    """
    try:
        import quark
        from quark.onnx import quantize
    except ImportError:
        print("❌ Error: AMD Quark not installed")
        print("   Install from: https://quark.docs.amd.com")
        print("   Or: pip install quark-ai (if available)")
        print("\n   Note: Quark is typically included with Ryzen AI Software")
        print("   Download from: https://ryzenai.docs.amd.com")
        print("\n" + "=" * 70)
        print("⚠ Quark requires AMD Ryzen AI Software (NPU required)")
        print("=" * 70)
        print("\n💡 ALTERNATIVES for systems without NPU:")
        print("   1. Use ONNX Runtime quantization (INT8):")
        print("      python quantize_onnx_model.py --input <model> --output <output> --method onnxruntime")
        print("   2. For INT4: Use PyTorch quantization tools (GPTQ, AutoRound) before ONNX export")
        print("   3. Use the model in PyTorch format with quantization (no ONNX conversion needed)")
        print("\n   Note: ONNX Runtime only supports INT8, not INT4.")
        print("   For INT4 quantization without NPU, quantize the PyTorch model first,")
        print("   then export to ONNX.")
        return False
    
    print("=" * 70)
    print("Quantizing with AMD Quark (for Ryzen AI NPU)")
    print("=" * 70)
    
    # Find ONNX model file(s)
    if os.path.isdir(model_path):
        onnx_files = list(Path(model_path).glob("*.onnx"))
        if not onnx_files:
            print(f"❌ No .onnx files found in {model_path}")
            return False
        model_file = str(max(onnx_files, key=lambda p: p.stat().st_size))
        print(f"Found ONNX model: {model_file}")
    else:
        model_file = model_path
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
    
    # Check if model is already quantized
    is_quantized, quant_type = check_if_model_is_quantized(model_file)
    if is_quantized:
        print(f"\n⚠ Warning: Input model appears to be already quantized ({quant_type})")
        if quantization_type == "int4" and quant_type == "int8":
            print("   Attempting to re-quantize INT8 → INT4...")
            print("   Note: This may not work. Best practice: Use original FP32/FP16 model.")
        elif quantization_type == quant_type:
            print(f"   Model is already {quantization_type}. Re-quantization may not be necessary.")
        print("   If this fails, use the original FP32/FP16 ONNX model instead.")
    
    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(output_path):
        quantized_model_path = os.path.join(output_path, os.path.basename(model_file))
    else:
        quantized_model_path = output_path
    
    try:
        print(f"\nQuantizing model with Quark...")
        print(f"  Input: {model_file}")
        print(f"  Output: {quantized_model_path}")
        print(f"  Quantization type: {quantization_type}")
        
        start_time = time.time()
        
        # Quark quantization API (adjust based on actual Quark API)
        # This is a placeholder - actual API may differ
        quantize(
            model_path=model_file,
            output_path=quantized_model_path,
            quantization_type=quantization_type,
            calibration_data=calibration_data,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Quantization completed in {elapsed:.2f} seconds")
        
        # Check file size
        if os.path.exists(quantized_model_path):
            size_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)
            original_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            reduction = (1 - size_mb / original_size_mb) * 100
            print(f"  Original size: {original_size_mb:.2f} MB")
            print(f"  Quantized size: {size_mb:.2f} MB")
            print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Quark quantization: {e}")
        print("\nNote: Quark API may differ. Check AMD Quark documentation:")
        print("  https://quark.docs.amd.com")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Method 2: ONNX Runtime Quantization (INT8 only - moved after Quark)
# ============================================================================

# (Function definition moved above for logical flow)

# ============================================================================
# Method 3: ONNX Neural Compressor
# ============================================================================

def quantize_with_neural_compressor(
    model_path: str,
    output_path: str,
    calibration_data: Optional[List[Dict[str, np.ndarray]]] = None
):
    """
    Quantize ONNX model using Intel Neural Compressor (also works for non-Intel hardware).
    
    Args:
        model_path: Path to input ONNX model file or directory
        output_path: Path to save quantized model
        calibration_data: Optional calibration data
    """
    try:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
    except ImportError:
        print("❌ Error: neural-compressor not installed")
        print("   Install with: pip install neural-compressor")
        return False
    
    print("=" * 70)
    print("Quantizing with ONNX Neural Compressor")
    print("=" * 70)
    
    # Find ONNX model file(s)
    if os.path.isdir(model_path):
        onnx_files = list(Path(model_path).glob("*.onnx"))
        if not onnx_files:
            print(f"❌ No .onnx files found in {model_path}")
            return False
        model_file = str(max(onnx_files, key=lambda p: p.stat().st_size))
        print(f"Found ONNX model: {model_file}")
    else:
        model_file = model_path
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
    
    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(output_path):
        quantized_model_path = os.path.join(output_path, os.path.basename(model_file))
    else:
        quantized_model_path = output_path
    
    try:
        print(f"\nQuantizing model with Neural Compressor...")
        print(f"  Input: {model_file}")
        print(f"  Output: {quantized_model_path}")
        
        # Create quantization config
        config = PostTrainingQuantConfig(
            approach="static",
            calibration_sampling_size=[100],  # Number of samples for calibration
        )
        
        # Create calibration dataset
        class CalibrationDataset:
            def __init__(self, data):
                self.data = data if data else []
            
            def __getitem__(self, idx):
                return self.data[idx % len(self.data)] if self.data else {}
            
            def __len__(self):
                return len(self.data) if self.data else 100
        
        calibration_dataset = CalibrationDataset(calibration_data)
        
        start_time = time.time()
        
        # Quantize the model
        quantized_model = quantization.fit(
            model=model_file,
            conf=config,
            calib_dataloader=calibration_dataset,
        )
        
        # Save quantized model
        quantized_model.save(quantized_model_path)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Quantization completed in {elapsed:.2f} seconds")
        
        # Check file size
        if os.path.exists(quantized_model_path):
            size_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)
            original_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            reduction = (1 - size_mb / original_size_mb) * 100
            print(f"  Original size: {original_size_mb:.2f} MB")
            print(f"  Quantized size: {size_mb:.2f} MB")
            print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Neural Compressor quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantize exported ONNX models for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize with AMD Quark (INT4 for NPU) - RECOMMENDED
  python quantize_onnx_model.py --input ./models/qwen3-coder-30b-onnx --output ./models/qwen3-coder-30b-onnx-int4

  # Quantize with AMD Quark (INT4) - explicit
  python quantize_onnx_model.py --input ./models/qwen3-coder-30b-onnx --method quark --quant-type int4 --output ./models/qwen3-coder-30b-onnx-int4

  # Quantize with ONNX Runtime (INT8 only - INT4 not supported)
  python quantize_onnx_model.py --input ./models/qwen3-coder-30b-onnx --method onnxruntime --output ./models/qwen3-coder-30b-onnx-int8

  # Quantize with Neural Compressor
  python quantize_onnx_model.py --input ./models/qwen3-coder-30b-onnx --method neural-compressor --output ./models/qwen3-coder-30b-onnx-int8-nc
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input ONNX model file or directory containing .onnx files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save quantized model (file or directory)"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["quark", "onnxruntime", "neural-compressor"],
        default="quark",
        help="Quantization method to use (default: quark for INT4 support)"
    )
    
    parser.add_argument(
        "--quant-type",
        type=str,
        choices=["int4", "int8", "bf16"],
        default="int4",
        help="Quantization type (default: int4 for best NPU performance)"
    )
    
    parser.add_argument(
        "--quant-format",
        type=str,
        choices=["QDQ", "QOperator"],
        default="QDQ",
        help="Quantization format for ONNX Runtime (QDQ or QOperator)"
    )
    
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to calibration data file (numpy .npz or pickle format)"
    )
    
    args = parser.parse_args()
    
    # Load calibration data if provided
    calibration_data = None
    if args.calibration_data:
        try:
            if args.calibration_data.endswith('.npz'):
                data = np.load(args.calibration_data)
                calibration_data = [dict(data)]
            else:
                import pickle
                with open(args.calibration_data, 'rb') as f:
                    calibration_data = pickle.load(f)
            print(f"✓ Loaded calibration data from {args.calibration_data}")
        except Exception as e:
            print(f"⚠ Warning: Could not load calibration data: {e}")
            print("   Continuing with dummy calibration data...")
    
    # Check if input model is already quantized
    input_model_path = args.input
    if os.path.isdir(input_model_path):
        onnx_files = list(Path(input_model_path).glob("*.onnx"))
        if onnx_files:
            input_model_path = str(max(onnx_files, key=lambda p: p.stat().st_size))
    
    if os.path.exists(input_model_path):
        is_quantized, existing_quant_type = check_if_model_is_quantized(input_model_path)
        if is_quantized:
            print("\n" + "=" * 70)
            print(f"⚠ Input Model Already Quantized ({existing_quant_type.upper()})")
            print("=" * 70)
            if args.quant_type == "int4" and existing_quant_type == "int8":
                print("\n💡 Converting INT8 → INT4:")
                print("   - AMD Quark may support this (will attempt)")
                print("   - Best practice: Use original FP32/FP16 model for better results")
                print("   - If Quark fails, you need the original FP32/FP16 ONNX model")
            elif args.quant_type == existing_quant_type:
                print(f"\n⚠ Model is already {args.quant_type.upper()}.")
                print("   Re-quantization may not be necessary or may fail.")
                print("   Consider using the original FP32/FP16 model instead.")
            print("=" * 70 + "\n")
    
    # Warn if trying to use INT4 with ONNX Runtime (not supported)
    if args.method == "onnxruntime" and args.quant_type == "int4":
        print("⚠ Warning: ONNX Runtime does not support INT4 quantization!")
        print("   Switching to INT8 quantization instead.")
        print("   For INT4 quantization, use: --method quark")
        args.quant_type = "int8"
    
    # Run quantization based on selected method
    success = False
    if args.method == "quark":
        success = quantize_with_quark(
            model_path=args.input,
            output_path=args.output,
            quantization_type=args.quant_type,
            calibration_data=calibration_data,
        )
        
        # If Quark failed and user wanted INT4, suggest ONNX Runtime as fallback
        if not success and args.quant_type == "int4":
            print("\n" + "=" * 70)
            print("💡 Fallback Option: Use ONNX Runtime (INT8)")
            print("=" * 70)
            print("Quark is not available (requires NPU/Ryzen AI Software).")
            print("You can use ONNX Runtime quantization with INT8 instead:")
            print(f"\n  python quantize_onnx_model.py \\")
            print(f"    --input {args.input} \\")
            print(f"    --output {args.output.replace('int4', 'int8')} \\")
            print(f"    --method onnxruntime")
            print("\nNote: ONNX Runtime only supports INT8, not INT4.")
            print("For INT4 without NPU, quantize the PyTorch model first, then export to ONNX.")
    elif args.method == "onnxruntime":
        success = quantize_with_onnxruntime(
            model_path=args.input,
            output_path=args.output,
            calibration_data=calibration_data,
            quant_format=args.quant_format,
        )
    elif args.method == "neural-compressor":
        success = quantize_with_neural_compressor(
            model_path=args.input,
            output_path=args.output,
            calibration_data=calibration_data,
        )
    
    if success:
        print("\n" + "=" * 70)
        print("✓ Quantization completed successfully!")
        print("=" * 70)
        print(f"\nQuantized model saved to: {os.path.abspath(args.output)}")
        print("\nNext steps:")
        print("1. Validate the quantized model with test inputs")
        print("2. For Ryzen AI NPU: Use VitisAIExecutionProvider in ONNX Runtime")
        print("3. Compare accuracy and performance with the original model")
    else:
        print("\n" + "=" * 70)
        print("❌ Quantization failed")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("1. Check that the input ONNX model is valid")
        print("2. Ensure all required dependencies are installed")
        print("3. For AMD Quark: Install Ryzen AI Software from AMD (requires NPU)")
        print("4. For systems without NPU: Use --method onnxruntime (INT8 only)")
        print("5. Provide representative calibration data for better accuracy")
        print("\n💡 Quick fix for Linux without NPU:")
        print(f"   python quantize_onnx_model.py --input {args.input} --output {args.output} --method onnxruntime")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

