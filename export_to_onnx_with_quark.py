"""
Script to export Qwen3-Coder-30B model to ONNX and quantize with AMD Quark
This script provides an end-to-end workflow for NPU deployment:
1. Load model from local folder
2. Export to ONNX format
3. Quantize with AMD Quark (INT4/BF16) for Ryzen AI NPU
4. Prepare for NPU deployment

Usage:
    python export_to_onnx_with_quark.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import signal
import resource
import time
import threading
import shutil

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install psutil for better memory monitoring: pip install psutil")

# ============================================================================
# Configuration
# ============================================================================

# Path to your local model folder
MODEL_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct"

# Quantization options
QUANTIZATION_TYPE = "int4"  # Options: "int4", "int8", "bf16"
# int4: Maximum compression, best for NPU memory constraints
# int8: Balanced compression and accuracy
# bf16: Best accuracy, supported for transformers on NPU

# Export options
SKIP_ONNX_EXPORT = False  # Set to True if you already have an ONNX model
# If True, will skip Step 2 and use existing ONNX model for quantization

# Output paths
ONNX_MODEL_PATH = "./models/qwen3-coder-30b-onnx"
QUANTIZED_MODEL_PATH = f"./models/qwen3-coder-30b-onnx-{QUANTIZATION_TYPE}"

# ============================================================================
# Multi-CPU Configuration - Use all CPUs and threads
# ============================================================================
def configure_multi_cpu():
    """Configure the script to use all available CPUs and threads"""
    import multiprocessing
    
    total_threads = multiprocessing.cpu_count()
    print(f"System CPU Configuration:")
    print(f"  Total CPU threads available: {total_threads}")
    
    # Set CPU affinity to use all CPUs (if possible)
    try:
        if HAS_PSUTIL:
            p = psutil.Process()
            available_cpus = list(range(total_threads))
            try:
                p.cpu_affinity(available_cpus)
                print(f"✓ CPU affinity set to use all {total_threads} threads")
            except (OSError, AttributeError) as e:
                print(f"  Note: Could not set CPU affinity: {e}")
    except Exception as e:
        print(f"  Note: Could not configure CPU affinity: {e}")
    
    # Configure PyTorch threading
    torch.set_num_threads(total_threads)
    torch.set_num_interop_threads(total_threads)
    print(f"✓ PyTorch configured to use {total_threads} threads")
    
    # Set environment variables for OpenMP, MKL, etc.
    os.environ['OMP_NUM_THREADS'] = str(total_threads)
    os.environ['MKL_NUM_THREADS'] = str(total_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(total_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(total_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(total_threads)
    print(f"✓ Environment variables set for {total_threads} threads")
    
    # Check for NUMA
    try:
        if HAS_PSUTIL:
            numa_nodes = psutil.cpu_count(logical=False)
            if numa_nodes and numa_nodes > 1:
                print(f"  Detected {numa_nodes} physical CPU sockets (NUMA system)")
    except Exception:
        pass
    
    print()

# ============================================================================
# OOM Protection and Memory Monitoring
# ============================================================================
def check_system_resources():
    """Check available system resources and warn if low"""
    if not HAS_PSUTIL:
        print("System Resources: (psutil not available)")
        return
    
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        mem_free_gb = mem.available / (1024**3)
        swap_free_gb = swap.free / (1024**3)
        
        print("System Resources:")
        print(f"  RAM: {mem_free_gb:.1f}GB free / {mem.total/(1024**3):.1f}GB total ({mem.percent}% used)")
        print(f"  Swap: {swap_free_gb:.1f}GB free / {swap.total/(1024**3):.1f}GB total ({swap.percent}% used)")
        
        if mem_free_gb < 16:
            print(f"\n⚠ WARNING: Low RAM ({mem_free_gb:.1f}GB free)")
            print("   The process may be killed by OOM killer if memory runs out")
        
        if swap_free_gb < 10:
            print(f"\n⚠ WARNING: Low swap space ({swap_free_gb:.1f}GB free)")
    except Exception as e:
        print(f"  (Could not check resources: {e})")

def adjust_oom_score():
    """Try to adjust OOM score to make process less likely to be killed"""
    try:
        pid = os.getpid()
        oom_score_adj_path = f"/proc/{pid}/oom_score_adj"
        if os.path.exists(oom_score_adj_path):
            try:
                with open(oom_score_adj_path, 'w') as f:
                    f.write("-500")
                print("✓ Adjusted OOM score (process less likely to be killed)")
            except PermissionError:
                print("⚠ Could not adjust OOM score (requires permissions)")
    except Exception:
        pass

def setup_signal_handlers():
    """Setup signal handlers to catch kill signals"""
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n\n⚠ Process received {sig_name} signal")
        print("Common causes: OOM killer, resource limits, or manual termination")
        raise SystemExit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# Main Script
# ============================================================================

# Configure multi-CPU support early
configure_multi_cpu()

# Check resources
check_system_resources()
print()

# Adjust OOM score
adjust_oom_score()
print()

# Setup signal handlers
setup_signal_handlers()

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {os.path.abspath(MODEL_PATH)}")
    print("Please update MODEL_PATH in the script")
    exit(1)

print("=" * 70)
print("AMD Quark ONNX Export and Quantization")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Quantization: {QUANTIZATION_TYPE.upper()}")
print(f"Output ONNX: {ONNX_MODEL_PATH}")
print(f"Output Quantized: {QUANTIZED_MODEL_PATH}")
print("=" * 70 + "\n")

# ============================================================================
# Step 1: Load Model (only if exporting to ONNX)
# ============================================================================
if not SKIP_ONNX_EXPORT:
    print("Step 1: Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cpu"},
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print("✓ Model loaded successfully!")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}\n")
else:
    print("Step 1: Skipped (using existing ONNX model)\n")

# ============================================================================
# Step 2: Export to ONNX (or use existing)
# ============================================================================
if SKIP_ONNX_EXPORT:
    print("=" * 70)
    print("Step 2: Using Existing ONNX Model")
    print("=" * 70)
    print(f"Using existing ONNX model at: {ONNX_MODEL_PATH}")
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ Error: ONNX model not found at {ONNX_MODEL_PATH}")
        print("   Set SKIP_ONNX_EXPORT=False to export from PyTorch model")
        exit(1)
    onnx_files = [f for f in os.listdir(ONNX_MODEL_PATH) if f.endswith('.onnx')]
    if onnx_files:
        total_size = sum(os.path.getsize(os.path.join(ONNX_MODEL_PATH, f)) 
                       for f in onnx_files) / (1024**3)
        print(f"✓ Found {len(onnx_files)} ONNX file(s), total size: {total_size:.2f} GB\n")
else:
    print("=" * 70)
    print("Step 2: Exporting to ONNX Format")
    print("=" * 70)

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError:
        print("Error: optimum[onnxruntime] not installed")
        print("Install with: pip install 'optimum[onnxruntime]'")
        exit(1)

    os.makedirs(ONNX_MODEL_PATH, exist_ok=True)

    # Fix for /tmp space issues
    import tempfile
    temp_dir = os.path.join(os.path.dirname(ONNX_MODEL_PATH), 'tmp_export')
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['TMPDIR'] = temp_dir
    os.environ['TMP'] = temp_dir
    os.environ['TEMP'] = temp_dir
    tempfile.tempdir = temp_dir
    print(f"✓ Using temporary directory: {temp_dir}\n")

    # Set ONNX opset
    os.environ['ONNX_OPSET'] = '17'

    print("Exporting model to ONNX (this may take hours for 30B models)...")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = time.time()

    try:
        ort_model = ORTModelForCausalLM.from_pretrained(
            MODEL_PATH,
            export=True,
            trust_remote_code=True
        )
        
        # Save the ONNX model
        print("\nSaving ONNX model files...")
        ort_model.save_pretrained(ONNX_MODEL_PATH)
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"\n✓ ONNX export completed in {hours}h {minutes}m")
        print(f"✓ Model saved to: {ONNX_MODEL_PATH}")
        
        # Check output size
        if os.path.exists(ONNX_MODEL_PATH):
            onnx_files = [f for f in os.listdir(ONNX_MODEL_PATH) if f.endswith('.onnx')]
            if onnx_files:
                total_size = sum(os.path.getsize(os.path.join(ONNX_MODEL_PATH, f)) 
                               for f in onnx_files) / (1024**3)
                print(f"  ONNX model size: {total_size:.2f} GB")
        
    except Exception as e:
        print(f"\n✗ Error during ONNX export: {e}")
        print("\nNote: You can skip ONNX export and quantize an existing ONNX model")
        print("      by setting SKIP_ONNX_EXPORT=True and providing ONNX_MODEL_PATH")
        raise

    # Clean up temp files
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

# ============================================================================
# Step 3: Quantize with AMD Quark
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Quantizing with AMD Quark")
print("=" * 70)

# Try multiple ways to import Quark
HAS_QUARK = False
quark_quantize_func = None

# Method 1: Try standard Quark import
try:
    import quark
    from quark.onnx import quantize
    quark_quantize_func = quantize
    HAS_QUARK = True
    print("✓ AMD Quark found (standard import)")
except ImportError:
    pass

# Method 2: Try amd-quark import (alternative package name)
if not HAS_QUARK:
    try:
        from amd_quark import quantize
        quark_quantize_func = quantize
        HAS_QUARK = True
        print("✓ AMD Quark found (amd-quark package)")
    except ImportError:
        pass

# Method 3: Try quark-ai package
if not HAS_QUARK:
    try:
        from quark_ai import quantize
        quark_quantize_func = quantize
        HAS_QUARK = True
        print("✓ AMD Quark found (quark-ai package)")
    except ImportError:
        pass

if not HAS_QUARK:
    print("⚠ AMD Quark not found")
    print("\nTo use AMD Quark quantization:")
    print("1. Install AMD Ryzen AI Software from: https://ryzenai.docs.amd.com")
    print("2. Quark is included with the Ryzen AI Software suite")
    print("3. Activate the Ryzen AI conda environment")
    print("4. Or try: pip install amd-quark (if available)")
    print("\nAlternative: Use ONNX Runtime quantization (see quantize_onnx_model.py)")

if HAS_QUARK:
    os.makedirs(QUANTIZED_MODEL_PATH, exist_ok=True)
    
    # Find the main ONNX model file
    onnx_files = [f for f in os.listdir(ONNX_MODEL_PATH) if f.endswith('.onnx')]
    if not onnx_files:
        print(f"❌ No .onnx files found in {ONNX_MODEL_PATH}")
        exit(1)
    
    # Use the largest file (usually the main model)
    main_onnx_file = max(onnx_files, 
                        key=lambda f: os.path.getsize(os.path.join(ONNX_MODEL_PATH, f)))
    input_model_path = os.path.join(ONNX_MODEL_PATH, main_onnx_file)
    output_model_path = os.path.join(QUANTIZED_MODEL_PATH, f"model_{QUANTIZATION_TYPE}.onnx")
    
    print(f"Input ONNX model: {input_model_path}")
    print(f"Output quantized model: {output_model_path}")
    print(f"Quantization type: {QUANTIZATION_TYPE.upper()}")
    print("\nQuantizing with AMD Quark...")
    print("(This may take 30-60 minutes for a 30B model)\n")
    
    start_time = time.time()
    
    try:
        # Quark quantization
        # Note: Actual API may vary - check AMD Quark documentation
        # Try different API signatures based on Quark version
        if quark_quantize_func:
            try:
                # Try standard API
                quark_quantize_func(
                    model_path=input_model_path,
                    output_path=output_model_path,
                    quantization_type=QUANTIZATION_TYPE,
                )
            except (TypeError, AttributeError) as e1:
                # Try alternative API signature
                try:
                    quark_quantize_func(
                        input_model_path,
                        output_model_path,
                        quant_type=QUANTIZATION_TYPE,
                    )
                except (TypeError, AttributeError) as e2:
                    # Try with dict parameters
                    try:
                        quark_quantize_func(
                            input_model_path,
                            output_model_path,
                            **{"quantization_type": QUANTIZATION_TYPE}
                        )
                    except Exception as e3:
                        print(f"❌ All Quark API attempts failed:")
                        print(f"   Attempt 1: {e1}")
                        print(f"   Attempt 2: {e2}")
                        print(f"   Attempt 3: {e3}")
                        print("\nPlease check AMD Quark documentation for correct API:")
                        print("  https://quark.docs.amd.com")
                        raise
        else:
            raise ImportError("Quark quantize function not available")
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"\n✓ Quantization completed in {hours}h {minutes}m")
        
        # Check file sizes
        if os.path.exists(output_model_path):
            quantized_size = os.path.getsize(output_model_path) / (1024**3)
            original_size = os.path.getsize(input_model_path) / (1024**3)
            reduction = (1 - quantized_size / original_size) * 100
            
            print(f"  Original size: {original_size:.2f} GB")
            print(f"  Quantized size: {quantized_size:.2f} GB")
            print(f"  Size reduction: {reduction:.1f}%")
            print(f"\n✓ Quantized model saved to: {QUANTIZED_MODEL_PATH}")
        
    except Exception as e:
        print(f"\n✗ Error during Quark quantization: {e}")
        print("\nNote: Quark API may differ. Check AMD Quark documentation:")
        print("  https://quark.docs.amd.com")
        print("  https://ryzenai.docs.amd.com")
        import traceback
        traceback.print_exc()
        print("\nYou can manually quantize using:")
        print(f"  python quantize_onnx_model.py --input {ONNX_MODEL_PATH} --method quark --quant-type {QUANTIZATION_TYPE}")
else:
    print("\n⚠ Skipping Quark quantization (not available)")
    print(f"  ONNX model is available at: {ONNX_MODEL_PATH}")
    print("  You can quantize it later using quantize_onnx_model.py")

# ============================================================================
# Step 4: Prepare for NPU Deployment
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: NPU Deployment Information")
print("=" * 70)

if HAS_QUARK and os.path.exists(output_model_path):
    print(f"✓ Quantized model ready for NPU: {QUANTIZED_MODEL_PATH}")
    print("\nNext steps:")
    print("1. Compile for NPU using Vitis AI Execution Provider:")
    print("   from onnxruntime import InferenceSession")
    print("   from onnxruntime.vitisai import VitisAIExecutionProvider")
    print("   ")
    print(f"   session = InferenceSession('{output_model_path}',")
    print("                               providers=['VitisAIExecutionProvider'])")
    print("\n2. Run inference on NPU")
    print("3. Monitor NPU utilization with Ryzen AI tools")
else:
    print("⚠ Quantized model not available")
    print(f"  ONNX model available at: {ONNX_MODEL_PATH}")
    print("  Quantize manually or install AMD Quark")

print("\n" + "=" * 70)
print("Export and Quantization Complete!")
print("=" * 70)

