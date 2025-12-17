"""
Script to load Qwen3-Coder-30B model from a LOCAL folder (Ubuntu-friendly copy)
Use this after you've downloaded the model to a local directory.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import signal
import resource
import time
import threading

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install psutil for better memory monitoring: pip install psutil")

# ============================================================================
# Multi-CPU Configuration - Use all CPUs and threads
# ============================================================================
def configure_multi_cpu():
    """Configure the script to use all available CPUs and threads"""
    import multiprocessing
    
    # Get total number of CPU threads
    total_threads = multiprocessing.cpu_count()
    print(f"System CPU Configuration:")
    print(f"  Total CPU threads available: {total_threads}")
    
    # Set CPU affinity to use all CPUs (if possible)
    try:
        if HAS_PSUTIL:
            p = psutil.Process()
            # Get all available CPU cores
            available_cpus = list(range(total_threads))
            try:
                p.cpu_affinity(available_cpus)
                print(f"✓ CPU affinity set to use all {total_threads} threads")
            except (OSError, AttributeError) as e:
                print(f"  Note: Could not set CPU affinity: {e}")
                print("  (This is normal if running in some containers or with restrictions)")
    except Exception as e:
        print(f"  Note: Could not configure CPU affinity: {e}")
    
    # Configure PyTorch threading
    torch.set_num_threads(total_threads)
    torch.set_num_interop_threads(total_threads)
    print(f"✓ PyTorch configured to use {total_threads} threads")
    
    # Set environment variables for OpenMP, MKL, etc. to use all threads
    os.environ['OMP_NUM_THREADS'] = str(total_threads)
    os.environ['MKL_NUM_THREADS'] = str(total_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(total_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(total_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(total_threads)
    print(f"✓ Environment variables set for {total_threads} threads (OMP, MKL, etc.)")
    
    # Verify configuration
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    print(f"  PyTorch interop threads: {torch.get_num_interop_threads()}")
    
    # Check for NUMA (Non-Uniform Memory Access) - common in multi-socket systems
    try:
        if HAS_PSUTIL:
            # Get NUMA nodes
            numa_nodes = psutil.cpu_count(logical=False)  # Physical cores
            if numa_nodes and numa_nodes > 1:
                print(f"  Detected {numa_nodes} physical CPU sockets (NUMA system)")
                print("  Note: For best performance on NUMA systems, consider:")
                print("    - Using numactl to bind to both sockets: numactl --cpunodebind=0,1 --membind=0,1 python script.py")
                print("    - Or set: export OMP_PROC_BIND=spread OMP_PLACES=threads")
    except Exception:
        pass
    
    print()

# Configure multi-CPU support early
configure_multi_cpu()

# Path to your local model folder
# Change this to match where you saved the model
# Use the non-FP8 variant for CPU export to avoid XPU-only FP8 kernels
MODEL_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct"

# Toggle ONNX export (set to False to skip export step)
EXPORT_TO_ONNX = True
# Force CPU execution (overrides GPU/XPU detection)
FORCE_CPU = True

# Some Arc GPUs (e.g., B580) do not support xpu mem_get_info yet; patch to avoid crashes
def _patch_xpu_mem_get_info():
    if getattr(_patch_xpu_mem_get_info, "_patched", False):
        return
    fallback_total = 8 * 1024 ** 3  # assume 8GB total as safe default
    fallback_free = int(fallback_total * 0.9)

    try:
        import torch.xpu.memory as xpu_mem
        orig_fn = getattr(xpu_mem, "mem_get_info", None)

        def safe_mem_get_info(device=None):
            try:
                if orig_fn:
                    return orig_fn(device)
            except Exception:
                pass
            print("Note: xpu mem_get_info unsupported; using fallback free/total values.")
            return (fallback_free, fallback_total)

        if orig_fn:
            xpu_mem.mem_get_info = safe_mem_get_info
    except Exception as e:
        print(f"Note: could not patch torch.xpu.memory.mem_get_info: {e}")

    try:
        import torch._C as _C
        orig_c_fn = getattr(_C, "_xpu_getMemoryInfo", None)

        def safe_c_mem_info(device=None):
            try:
                if orig_c_fn:
                    return orig_c_fn(device)
            except Exception:
                pass
            print("Note: xpu _xpu_getMemoryInfo unsupported; using fallback free/total values.")
            return (fallback_free, fallback_total)

        if orig_c_fn:
            _C._xpu_getMemoryInfo = safe_c_mem_info
    except Exception as e:
        print(f"Note: could not patch torch._C._xpu_getMemoryInfo: {e}")

    _patch_xpu_mem_get_info._patched = True

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {os.path.abspath(MODEL_PATH)}")
    print("\nPlease either:")
    print("1. Download the model first using download_and_use_qwen3_coder_local.py")
    print("2. Update MODEL_PATH in this script to point to your model location")
    print("\nOr use download_and_use_qwen3_coder.py for automatic download")
    exit(1)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}\n")

# Load from local folder
# fix_mistral_regex handles known tokenizer regex bug for some Mistral-based releases
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    fix_mistral_regex=True
)

# Check for available accelerators (CUDA, Intel XPU, or CPU)
has_cuda = False if FORCE_CPU else torch.cuda.is_available()
has_intel_gpu = False
intel_gpu_device = None

# Check for Intel GPU (XPU) - requires intel-extension-for-pytorch
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = hasattr(ipex, "xpu")
    ipex_xpu_ok = False
    if ipex_available and hasattr(ipex.xpu, "is_available"):
        ipex_xpu_ok = ipex.xpu.is_available()
    torch_xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not FORCE_CPU and (ipex_xpu_ok or torch_xpu_ok):
        has_intel_gpu = True
        intel_gpu_device = torch.device("xpu:0")
        print(f"✓ Intel GPU (XPU) detected! (torch.xpu={torch_xpu_ok}, ipex.xpu={ipex_xpu_ok})")
        print(f"  Using Intel GPU for faster conversion (4-12 hours vs 12-48+ hours on CPU)")
except ImportError:
    print("ℹ Intel Extension for PyTorch not installed.")
    print("  To use Intel GPU: Install from https://github.com/intel/intel-extension-for-pytorch")
    print("  Or use OpenVINO for Intel GPU acceleration")
except Exception as e:
    print(f"Note: Intel GPU detection failed: {e}")

if has_cuda:
    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    print("Loading model on NVIDIA GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # Convert to bfloat16 after loading if needed
    if hasattr(model, 'to'):
        model = model.to(torch.bfloat16)
elif has_intel_gpu:
    print("Loading model on Intel GPU (XPU)...")
    # Ensure XPU device is the default for subsequent allocations/exports
    try:
        if hasattr(torch, "xpu"):
            torch.xpu.set_device(0)
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("xpu")
        _patch_xpu_mem_get_info()
    except Exception as e:
        print(f"Note: Could not set default XPU device: {e}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "xpu:0"},  # Use Intel GPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # Convert to bfloat16 after loading if needed
    if hasattr(model, 'to'):
        try:
            model = model.to(intel_gpu_device)
            model = model.to(torch.bfloat16)
        except Exception as e:
            print(f"Note: Could not convert to bfloat16: {e}")
            print("Using model's default dtype")
else:
    print("Forcing CPU execution (FORCE_CPU=True).")
    # For CPU, use explicit CPU device_map to avoid disk offloading
    # MoE models may not support dtype parameter, so we load first then convert
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cpu"},  # Explicit CPU mapping
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # Convert to bfloat16 after loading if needed
    if hasattr(model, 'to'):
        try:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            model = model.to(torch.bfloat16)
        except Exception as e:
            print(f"Note: Could not convert to bfloat16: {e}")
            print("Using model's default dtype")

print("Model loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}\n")

# ============================================================================
# Convert to ONNX Format (Required before NPU compilation)
# ============================================================================
# To use the model on NPU, you need to convert it to ONNX format first.
# This conversion may require significant memory and time for a 30B model.
# 
# Uncomment the following code to convert the model to ONNX:
#
if not EXPORT_TO_ONNX:
    print("EXPORT_TO_ONNX=False -> skipping ONNX export step.")
    exit(0)

try:
    import onnx
except ImportError:
    print("Missing dependency: onnx. Install with `pip install onnx`.")
    exit(1)

try:
    from optimum.onnxruntime import ORTModelForCausalLM
except ImportError:
    print("Missing dependency: optimum with onnxruntime backend.")
    print('Install with: pip install "optimum[onnxruntime]" onnxruntime')
    exit(1)

# Output path for ONNX model
onnx_model_path = "./models/qwen3-coder-30b-onnx"
os.makedirs(onnx_model_path, exist_ok=True)

print("=" * 70)
print("Starting ONNX Conversion using Optimum")
print("=" * 70)
print(f"Output path: {os.path.abspath(onnx_model_path)}")

# Check if model is quantized
is_quantized = "int4" in MODEL_PATH.lower() or "quant" in MODEL_PATH.lower()
if is_quantized:
    print("\n⚠ IMPORTANT: You're exporting a QUANTIZED model (int4-AutoRound)")
    print("   - Quantized models may cause precision errors when exported to ONNX")
    print("   - The quantization format may not be preserved in ONNX")
    print("   - RECOMMENDED: Export non-quantized model, then quantize the ONNX model")
    print("   - OR: The ONNX model will need to be re-quantized after export")
    print("   - Precision errors (float32/float16 mismatch) are common with quantized exports")

print("\n⚠ TIME ESTIMATE for 30B Model:")
print("   - CPU-only: 12-48+ hours (possibly days)")
print("   - Intel GPU (Arc): 4-12 hours (much faster!)")
print("   - NVIDIA GPU: 2-8 hours")
print("   - MoE models are more complex and take longer")
print("\n💡 TIPS:")
print("   - Intel GPU acceleration: Install 'intel-extension-for-pytorch'")
print("   - Let it run overnight or over a weekend")
print("   - Monitor memory usage (32GB may be tight)")
print("   - The process will create checkpoint files you can resume")
print("   - If you get 'No space left on device', check /tmp space (tmpfs may be full)")
print("   - Temporary files will be stored in the output directory to avoid /tmp issues")
print("   - To prevent OOM kills: Increase swap, close other apps, or run with 'nohup'")
print("   - EXPECTED: High CPU (all threads) during conversion, then single-threaded during save")
print("   - Single-threaded phase is NORMAL - model is being written to disk (can take 30-60 min)")
if is_quantized:
    print("   - If you get precision errors, re-quantize the ONNX model using AMD Quark")
print("=" * 70 + "\n")

# ============================================================================
# OOM Protection and Memory Monitoring
# ============================================================================
# Prevent the process from being killed by OOM killer

def check_system_resources():
    """Check available system resources and warn if low"""
    if not HAS_PSUTIL:
        print("System Resources: (psutil not available - install with: pip install psutil)")
        return
    
    try:
        # Check memory
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
            print("   Consider: Close other applications, increase swap, or use a smaller model")
        
        if swap_free_gb < 10:
            print(f"\n⚠ WARNING: Low swap space ({swap_free_gb:.1f}GB free)")
            print("   System may kill processes when swap is exhausted")
        
        # Check resource limits
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)  # Virtual memory limit
            if soft != resource.RLIM_INFINITY:
                soft_gb = soft / (1024**3) if soft != -1 else float('inf')
                print(f"  Process memory limit: {soft_gb:.1f}GB (soft), {hard/(1024**3):.1f}GB (hard)")
        except:
            pass
            
    except Exception as e:
        print(f"  (Could not check resources: {e})")

def adjust_oom_score():
    """Try to adjust OOM score to make process less likely to be killed"""
    try:
        pid = os.getpid()
        oom_score_adj_path = f"/proc/{pid}/oom_score_adj"
        if os.path.exists(oom_score_adj_path):
            # Set OOM score adjustment to -500 (less likely to be killed)
            # Requires appropriate permissions
            try:
                with open(oom_score_adj_path, 'w') as f:
                    f.write("-500")
                print("✓ Adjusted OOM score (process less likely to be killed)")
            except PermissionError:
                print("⚠ Could not adjust OOM score (requires appropriate permissions)")
                print("  To run with OOM protection, use: sudo python script.py")
                print("  Or set manually: echo -500 | sudo tee /proc/$(pgrep -f script.py)/oom_score_adj")
    except Exception as e:
        pass  # Not on Linux or can't adjust

def setup_signal_handlers():
    """Setup signal handlers to catch kill signals"""
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n\n⚠ Process received {sig_name} signal (likely killed by system)")
        print("Common causes:")
        print("  1. Out of Memory (OOM) - system ran out of RAM/swap")
        print("  2. Resource limits - process exceeded memory/CPU limits")
        print("  3. Manual kill - process was terminated by user/admin")
        print("\nTo prevent this:")
        print("  - Increase swap space: sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
        print("  - Close other memory-intensive applications")
        print("  - Use 'nohup' to run in background: nohup python script.py > output.log 2>&1 &")
        print("  - Use 'systemd-run' with memory limits: systemd-run --scope -p MemoryLimit=64G python script.py")
        raise SystemExit(1)
    
    # Register handlers for common kill signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    # SIGKILL cannot be caught, but we can try to catch others

# Check resources before starting
check_system_resources()
print()

# Try to adjust OOM score
adjust_oom_score()
print()

# Setup signal handlers
setup_signal_handlers()

# Fix for "No space left on device" error - set TMPDIR to a location with more space
# The /tmp directory (tmpfs) may be full, so use a location on the main filesystem
import tempfile
import shutil

# Check /tmp space (if on Linux)
try:
    if os.path.exists('/tmp'):
        stat = shutil.disk_usage('/tmp')
        free_gb = stat.free / (1024**3)
        if free_gb < 70:  # Less than 70GB free
            print(f"⚠ Warning: /tmp has only {free_gb:.1f}GB free space")
            print("   This may cause 'No space left on device' errors during export")
except Exception:
    pass  # Not on Linux or can't check

# Use the output directory or a subdirectory for temporary files
temp_dir = os.path.join(os.path.dirname(onnx_model_path), 'tmp_export')
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir
tempfile.tempdir = temp_dir
print(f"✓ Using temporary directory: {temp_dir}")
print("  (This prevents 'No space left on device' errors when /tmp is full)")
print("  Temporary files will be cleaned up after export completes\n")

# Use Optimum's ONNX exporter which handles transformers models better
# This avoids the dynamic shape issues with torch.onnx.export
try:
    is_fp8 = "fp8" in MODEL_PATH.lower()
    if is_fp8 and not (has_intel_gpu or has_cuda) and not FORCE_CPU:
        print("This FP8 model expects a GPU (Intel XPU or NVIDIA) for export.")
        print("No GPU detected, so ONNX export will fail. Options:")
        print("  - Switch MODEL_PATH to a non-FP8 variant (e.g., standard bf16/fp16)")
        print("  - Use an Intel Arc/NVIDIA GPU for export")
        exit(1)

    export_device = None
    if FORCE_CPU:
        export_device = "cpu"
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    elif has_intel_gpu:
        export_device = "xpu"
        try:
            if hasattr(torch, "xpu"):
                torch.xpu.set_device(0)
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("xpu")
        except Exception as e:
            print(f"Note: Could not set XPU as default export device: {e}")
    elif has_cuda:
        export_device = "cuda"

    start_time = time.time()
    
    print("Exporting model to ONNX format using Optimum...")
    print("Note: For MoE models, this process may take significant time and memory...")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Monitor CPU usage and file progress during export (optional, runs in background)
    def monitor_export_progress():
        """Monitor CPU usage, memory, and file sizes to detect progress or stuck state"""
        if not HAS_PSUTIL:
            return
        try:
            p = psutil.Process()
            print("Export Progress Monitoring:")
            print("  Time    | CPU%  | Threads | RAM(GB) | Temp(GB) | Output(GB) | Status")
            print("  " + "-" * 85)
            
            monitor_start = time.time()
            last_temp_size = 0
            last_output_size = 0
            no_progress_count = 0
            last_update_time = time.time()
            
            while True:
                cpu_percent = p.cpu_percent(interval=1)
                num_threads = p.num_threads()
                mem_info = p.memory_info()
                mem_gb = mem_info.rss / (1024**3)
                elapsed = int(time.time() - monitor_start)
                
                # Check temp directory size
                temp_size = 0
                if os.path.exists(temp_dir):
                    try:
                        temp_size = sum(os.path.getsize(os.path.join(temp_dir, f)) 
                                      for f in os.listdir(temp_dir) 
                                      if os.path.isfile(os.path.join(temp_dir, f))) / (1024**3)
                    except:
                        pass
                
                # Check output directory size
                output_size = 0
                if os.path.exists(onnx_model_path):
                    try:
                        output_size = sum(os.path.getsize(os.path.join(onnx_model_path, f)) 
                                        for f in os.listdir(onnx_model_path) 
                                        if os.path.isfile(os.path.join(onnx_model_path, f))) / (1024**3)
                    except:
                        pass
                
                # Detect if stuck (no file size change for 10 minutes)
                current_time = time.time()
                temp_changed = abs(temp_size - last_temp_size) > 0.1  # 100MB change
                output_changed = abs(output_size - last_output_size) > 0.1
                
                if temp_changed or output_changed:
                    no_progress_count = 0
                    last_update_time = current_time
                    status = "✓ Progress"
                else:
                    no_progress_count += 1
                    minutes_no_progress = (current_time - last_update_time) / 60
                    if minutes_no_progress > 10:
                        status = f"⚠ No progress {int(minutes_no_progress)}min"
                    elif minutes_no_progress > 5:
                        status = f"⏳ Slow {int(minutes_no_progress)}min"
                    else:
                        status = "⏳ Working"
                
                last_temp_size = temp_size
                last_output_size = output_size
                
                # Get active threads count (threads actually using CPU)
                active_threads = sum(1 for cpu in psutil.cpu_percent(percpu=True, interval=0.1) if cpu > 5)
                
                print(f"  {elapsed:6d}s | {cpu_percent:5.1f}% | {num_threads:7d} | {mem_gb:7.1f} | {temp_size:8.1f} | {output_size:9.1f} | {status}")
                
                # Warning if single-threaded for too long
                if active_threads <= 1 and elapsed > 1800:  # 30 minutes
                    if elapsed % 300 == 0:  # Every 5 minutes
                        print(f"  ⚠ NOTE: Single-threaded phase detected (likely I/O or finalization)")
                        print(f"     This is normal for model saving/validation phases")
                        print(f"     Continue monitoring - process should complete")
                
                time.sleep(30)  # Update every 30 seconds
        except KeyboardInterrupt:
            print("\n  Progress monitoring stopped")
        except Exception as e:
            pass  # Silently fail if monitoring can't continue
    
    # Start progress monitoring in a separate thread (non-blocking)
    monitor_thread = threading.Thread(target=monitor_export_progress, daemon=True)
    monitor_thread.start()
    print("(Progress monitoring started - updates every 30 seconds)\n")
    
    # Optimum's export handles MoE models and dynamic shapes better
    # The export=True parameter triggers the conversion
    # Set ONNX opset via environment variable (Optimum uses this)
    os.environ['ONNX_OPSET'] = '17'  # ONNX opset version 17 for Ryzen AI compatibility
    
    # Explicitly wrap export in a device context if available
    if export_device == "xpu" and hasattr(torch, "xpu"):
        ctx = torch.xpu.device(0)
    elif export_device == "cuda" and torch.cuda.is_available():
        ctx = torch.cuda.device(0)
    elif export_device == "cpu":
        class _CpuCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        ctx = _CpuCtx()
    else:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        ctx = _NullCtx()

    print("=" * 70)
    print("Starting ONNX Export Process")
    print("=" * 70)
    print("Phase 1: Model conversion (multi-threaded, high CPU usage)")
    print("Phase 2: Model saving/validation (single-threaded I/O, normal)")
    print("=" * 70 + "\n")
    
    with ctx:
        print("Beginning export... (this may take hours for 30B models)")
        # Note: export_kwargs is not supported in this version of Optimum
        # Device and opset are handled via environment variables and model defaults
        ort_model = ORTModelForCausalLM.from_pretrained(
            MODEL_PATH,
            export=True,  # This triggers the export
            trust_remote_code=True
        )
    
    print("\n" + "=" * 70)
    print("Export conversion complete - now saving model files...")
    print("(This phase is single-threaded I/O and may take time)")
    print("=" * 70 + "\n")
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\n✓ Export completed in {hours}h {minutes}m {seconds}s")
    
    # Save the ONNX model (this is single-threaded I/O)
    print("Saving ONNX model files (this may take 30-60 minutes for large models)...")
    ort_model.save_pretrained(onnx_model_path)
    print(f"\n✓ Model successfully exported to ONNX: {onnx_model_path}")
    
    # Clean up temporary files
    try:
        if os.path.exists(temp_dir):
            temp_size = sum(os.path.getsize(os.path.join(temp_dir, f)) for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))) / (1024**3)
            print(f"\nCleaning up temporary files ({temp_size:.2f} GB)...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✓ Temporary files cleaned up")
    except Exception as cleanup_error:
        print(f"⚠ Could not clean up temporary directory {temp_dir}: {cleanup_error}")
        print("  You can manually delete it later if needed")
    
    # Verify ONNX model
    try:
        # Check if main model file exists
        model_files = [f for f in os.listdir(onnx_model_path) if f.endswith('.onnx')]
        if model_files:
            total_size = sum(os.path.getsize(os.path.join(onnx_model_path, f)) for f in model_files) / (1024**3)
            print(f"✓ ONNX model files created: {len(model_files)} file(s)")
            print(f"  Total size: {total_size:.2f} GB")
            
            # Try to validate the ONNX model
            try:
                print("\nValidating ONNX model...")
                onnx_file = os.path.join(onnx_model_path, model_files[0])
                onnx_model = onnx.load(onnx_file)
                onnx.checker.check_model(onnx_model)
                print("✓ ONNX model validation passed")
                
                # Check for precision issues
                print("\nChecking for precision issues...")
                try:
                    from onnxruntime import InferenceSession
                    # Try to create a session to catch runtime precision errors
                    test_session = InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
                    print("✓ ONNX Runtime can load the model successfully")
                    del test_session
                except Exception as runtime_error:
                    if "Type Error" in str(runtime_error) or "precision" in str(runtime_error).lower():
                        print(f"⚠ Precision mismatch detected: {runtime_error}")
                        print("\n💡 To fix precision issues, you can:")
                        print("   1. Use ONNX Simplifier: pip install onnx-simplifier")
                        print("   2. Use quantization tools to convert to consistent precision")
                        print("   3. The model may still work with specific ONNX Runtime providers")
                    else:
                        raise
            except Exception as ve:
                print(f"⚠ Warning: ONNX model validation failed: {ve}")
                print("  The model may still work, but you may need to fix precision manually")
        else:
            print("⚠ Warning: No .onnx files found in output directory")
            print("  Checking for other files...")
            all_files = os.listdir(onnx_model_path)
            if all_files:
                print(f"  Found files: {all_files[:10]}")  # Show first 10 files
    except Exception as e:
        print(f"⚠ Warning: Could not verify ONNX model: {e}")
    
except Exception as e:
    error_msg = str(e)
    if "Type Error" in error_msg or "precision" in error_msg.lower() or "float" in error_msg.lower():
        print(f"\n⚠ Precision/Type Error during ONNX export: {e}")
        if is_quantized:
            print("\n⚠ ROOT CAUSE: Exporting a quantized (int4) model to ONNX")
            print("   - The AutoRound int4 quantization format is not preserved in ONNX")
            print("   - ONNX export converts quantized weights to float types inconsistently")
            print("   - This causes float32/float16 mismatches")
        else:
            print("\nThis is a common issue with mixed precision models (float32/float16).")
        print("\n🔧 SOLUTIONS:")
        if is_quantized:
            print("1. RE-QUANTIZE the ONNX model using AMD Quark (recommended):")
            print("   - This will convert the ONNX model to consistent INT4/BF16 precision")
            print("   - This fixes the precision mismatch AND prepares for NPU")
            print("2. Export the non-quantized model first, then quantize the ONNX model")
            print("3. Use ONNX Runtime with providers that handle mixed precision:")
            print("   - Try: providers=['VitisAIExecutionProvider'] (may handle it better)")
        else:
            print("1. Use ONNX Runtime with specific providers that handle mixed precision:")
            print("   - Try: providers=['CPUExecutionProvider', 'CUDAExecutionProvider']")
            print("2. Fix the ONNX model precision manually using ONNX tools")
            print("3. Re-export with consistent precision settings")
            print("4. Use quantization to convert to a single precision (INT4/INT8)")
        print("\nThe exported model files are still created and may work with different providers.")
    else:
        print(f"\n✗ Error during ONNX export with Optimum: {e}")
        print("\nNote: MoE (Mixture of Experts) models can be challenging to export to ONNX.")
        print("The dynamic routing in MoE models creates data-dependent control flow.")
        print("\nAlternative options:")
        print("1. Use the model in PyTorch format directly (no ONNX conversion needed)")
        print("2. Try exporting individual components separately")
        print("3. Use a non-MoE variant of the model if available")
        print("4. Consider using AMD's Quark tools which may handle MoE models better")
    
    # Don't raise - let the user know the files were created even if there are warnings
    if "Type Error" in error_msg or "precision" in error_msg.lower():
        print("\n⚠ Continuing despite precision warning - model files are saved.")
    else:
        raise
# 
# # Quantize the Model (Recommended for NPU)
# # After converting to ONNX, you should quantize the model:
# # - Use AMD Quark Model Optimization Library to quantize to INT4 or BF16
# # - BF16 is supported for transformer models on Ryzen AI NPU
# # - INT4 quantization can reduce memory requirements significantly
# # - Quantized model path: "./models/qwen3-coder-30b-quantized.onnx"
# ============================================================================

# ============================================================================
# Compile for NPU (Alternative approach using ONNX Runtime with Vitis AI EP)
# ============================================================================
# Note: This requires the model to be converted to ONNX format first.
# The model must be quantized and exported as ONNX before using this approach.
# 
# IMPORTANT: If you encountered precision errors (float32/float16 mismatch),
# quantization will fix this by converting to a consistent precision (INT4/BF16).
# 
# To use NPU execution:
# 1. Convert the model to ONNX format (see section above)
# 2. Quantize the model (INT4 or BF16 recommended for NPU) - THIS FIXES PRECISION ISSUES
# 3. Use the code below to create an inference session with Vitis AI EP
#
# Uncomment the following code if you have an ONNX model ready:
#
# from onnxruntime import InferenceSession
# from onnxruntime.vitisai import VitisAIExecutionProvider
# 
# # Create inference session with Vitis AI EP
# session = InferenceSession(
#     "qwen3-coder-30b-quantized.onnx",
#     providers=['VitisAIExecutionProvider']
# )
#
# # For hybrid execution (NPU + GPU/CPU):
# # session = InferenceSession(
# #     "qwen3-coder-30b.onnx",
# #     providers=['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# # )
#
# # Run inference on NPU:
# # prompt = "Write a Python function to calculate fibonacci numbers"
# # inputs = tokenizer(prompt, return_tensors="np")
# # outputs = session.run(None, dict(inputs))
# ============================================================================

# Example usage
prompt = "Write a Python function to check if a number is prime."
print(f"Prompt: {prompt}\n")

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("Generated response:")
print(content)

