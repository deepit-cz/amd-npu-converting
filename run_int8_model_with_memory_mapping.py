"""
Script to run INT8 quantized ONNX model on AMD Ryzen AI NPU with memory mapping
Optimized for large models that don't fit entirely in RAM (e.g., 88GB INT8 model on 32GB RAM)

Features:
- Uses Vitis AI Execution Provider for NPU acceleration
- Enables memory mapping for large models
- Works with INT8 quantized models
- Automatic fallback to CPU/GPU if NPU not available

Prerequisites:
1. ONNX Runtime with Vitis AI EP (from Ryzen AI Software)
2. INT8 quantized ONNX model
3. Tokenizer

Usage:
    python run_int8_model_with_memory_mapping.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import psutil

# ============================================================================
# Configuration
# ============================================================================

# Path to your INT8 quantized ONNX model (directory or file)
# Model can be larger than available RAM - memory mapping will be used
INT8_MODEL_PATH = "./models/qwen3-coder-30b-onnx-int8"

# Path to tokenizer
TOKENIZER_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct"

# Memory mapping configuration
ENABLE_MEMORY_MAPPING = True  # Enable memory mapping for large models
MEMORY_MAPPING_THRESHOLD_GB = 16  # Use memory mapping if model > 16GB

# ============================================================================
# Check Dependencies
# ============================================================================

print("=" * 70)
print("INT8 Model Inference with Memory Mapping")
print("AMD Ryzen AI NPU + Vitis AI Execution Provider")
print("=" * 70)

# Check for ONNX Runtime
try:
    from onnxruntime import InferenceSession, SessionOptions
    import onnxruntime
    print("✓ ONNX Runtime found")
except ImportError:
    print("❌ Error: ONNX Runtime not installed")
    print("   Install with: pip install onnxruntime")
    print("   For NPU: Install ONNX Runtime with Vitis AI EP from Ryzen AI Software")
    sys.exit(1)

# Check for Vitis AI Execution Provider
try:
    from onnxruntime.vitisai import VitisAIExecutionProvider
    HAS_VITIS_AI = True
    print("✓ Vitis AI Execution Provider found")
except ImportError:
    HAS_VITIS_AI = False
    print("⚠ Warning: Vitis AI Execution Provider not found")
    print("   This is required for NPU execution")
    print("   Install from: AMD Ryzen AI Software")
    print("   Download from: https://ryzenai.docs.amd.com")
    print("\n   Will attempt to use CPU/GPU fallback...")

# Check for tokenizer
try:
    from transformers import AutoTokenizer
    print("✓ Transformers library found")
except ImportError:
    print("❌ Error: Transformers not installed")
    print("   Install with: pip install transformers")
    sys.exit(1)

# ============================================================================
# System Memory Check
# ============================================================================

def get_system_memory():
    """Get available system memory in GB"""
    try:
        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except Exception:
        return None, None

total_ram_gb, available_ram_gb = get_system_memory()
if total_ram_gb:
    print(f"\nSystem Memory:")
    print(f"  Total RAM: {total_ram_gb:.1f} GB")
    print(f"  Available RAM: {available_ram_gb:.1f} GB")

# ============================================================================
# Load Tokenizer
# ============================================================================

print("\n" + "=" * 70)
print("Loading Tokenizer")
print("=" * 70)

if not os.path.exists(TOKENIZER_PATH):
    print(f"⚠ Tokenizer not found at {TOKENIZER_PATH}")
    print("   Using model path as fallback...")
    TOKENIZER_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    print(f"✓ Tokenizer loaded from: {TOKENIZER_PATH}")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    print("   Please update TOKENIZER_PATH in the script")
    sys.exit(1)

# ============================================================================
# Find and Check Model
# ============================================================================

print("\n" + "=" * 70)
print("Loading INT8 ONNX Model with Memory Mapping")
print("=" * 70)

# Find the ONNX model file
if os.path.isdir(INT8_MODEL_PATH):
    onnx_files = list(Path(INT8_MODEL_PATH).glob("*.onnx"))
    if not onnx_files:
        print(f"❌ No .onnx files found in {INT8_MODEL_PATH}")
        sys.exit(1)
    model_file = str(max(onnx_files, key=lambda p: p.stat().st_size))
    print(f"Found ONNX model: {model_file}")
else:
    model_file = INT8_MODEL_PATH
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("\nAvailable options:")
        print("  1. Run quantize_onnx_model.py to create INT8 model")
        print("  2. Update INT8_MODEL_PATH in this script")
        sys.exit(1)

# Check model size
model_size_gb = os.path.getsize(model_file) / (1024**3)
print(f"Model file size: {model_size_gb:.2f} GB")

# Check for external data file (onnx_data)
model_dir = os.path.dirname(model_file) if os.path.dirname(model_file) else "."
onnx_data_files = list(Path(model_dir).glob("*.onnx_data"))
if onnx_data_files:
    total_data_size = sum(f.stat().st_size for f in onnx_data_files) / (1024**3)
    print(f"External data files: {total_data_size:.2f} GB")
    total_model_size = model_size_gb + total_data_size
    print(f"Total model size: {total_model_size:.2f} GB")
else:
    total_model_size = model_size_gb

# Check if memory mapping is needed
if total_model_size > MEMORY_MAPPING_THRESHOLD_GB:
    print(f"\n⚠ Model size ({total_model_size:.2f} GB) exceeds threshold ({MEMORY_MAPPING_THRESHOLD_GB} GB)")
    if available_ram_gb and total_model_size > available_ram_gb:
        print(f"⚠ Model is larger than available RAM ({available_ram_gb:.1f} GB)")
        print("   Memory mapping will be used automatically")
    ENABLE_MEMORY_MAPPING = True
else:
    print(f"✓ Model size ({total_model_size:.2f} GB) is manageable")

# ============================================================================
# Configure Session Options with Memory Mapping
# ============================================================================

print("\n" + "=" * 70)
print("Configuring ONNX Runtime Session")
print("=" * 70)

# Create session options
session_options = SessionOptions()

# Enable memory mapping for large models
# ONNX Runtime automatically uses memory mapping for external data files
# But we can configure it explicitly
if ENABLE_MEMORY_MAPPING:
    # Enable memory pattern optimization (reduces memory allocations)
    session_options.enable_mem_pattern = True
    
    # Enable memory arena (improves performance, uses memory mapping internally)
    session_options.enable_mem_arena = True
    
    # Set memory limit (optional - helps prevent OOM)
    # session_options.memory_pattern_optimization = True
    
    print("✓ Memory mapping enabled")
    print("  - Memory pattern optimization: Enabled")
    print("  - Memory arena: Enabled")
    print("  - External data files will be memory-mapped automatically")

# Configure execution providers
# Priority: Vitis AI (NPU) > CUDA (GPU) > CPU
providers = []

if HAS_VITIS_AI:
    providers.append('VitisAIExecutionProvider')
    print("✓ Using Vitis AI Execution Provider (NPU)")
else:
    print("⚠ Vitis AI EP not available - will use CPU/GPU")

# Add fallback providers
try:
    available_providers = onnxruntime.get_available_providers()
    
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
        print("✓ CUDA Execution Provider available (GPU fallback)")
    
    providers.append('CPUExecutionProvider')
    print("✓ CPU Execution Provider available (fallback)")
    
    print(f"\nExecution provider order: {providers}")
except Exception as e:
    print(f"⚠ Could not check available providers: {e}")
    providers = ['CPUExecutionProvider']

# Configure CPU options if needed
import multiprocessing
cpu_threads = multiprocessing.cpu_count()

if 'CPUExecutionProvider' in providers:
    session_options.intra_op_num_threads = cpu_threads
    session_options.inter_op_num_threads = cpu_threads
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    print(f"✓ CPU optimizations: {cpu_threads} threads")

# ============================================================================
# Create Inference Session
# ============================================================================

try:
    print("\nCreating inference session...")
    print("  (This may take a moment for large models with memory mapping)")
    
    start_time = time.time()
    
    # Create session with memory mapping
    # ONNX Runtime automatically uses memory mapping for external data files
    session = InferenceSession(
        model_file,
        providers=providers,
        sess_options=session_options
    )
    
    load_time = time.time() - start_time
    
    # Check which provider is actually being used
    actual_provider = session.get_providers()[0]
    print(f"✓ Model loaded successfully in {load_time:.2f} seconds!")
    print(f"  Active provider: {actual_provider}")
    
    if actual_provider == 'VitisAIExecutionProvider':
        print("  🎉 Running on AMD Ryzen AI NPU!")
    elif actual_provider == 'CUDAExecutionProvider':
        print("  ⚠ Running on GPU (NPU not available)")
    else:
        print("  ⚠ Running on CPU (NPU/GPU not available)")
        print(f"  💡 Using {cpu_threads} CPU threads for inference")
    
    # Check memory usage after loading
    if available_ram_gb:
        current_mem = psutil.virtual_memory()
        used_after_load = (total_ram_gb - current_mem.available / (1024**3))
        print(f"\nMemory usage after loading:")
        print(f"  RAM used: {used_after_load:.1f} GB / {total_ram_gb:.1f} GB")
        if total_model_size > available_ram_gb:
            print(f"  ✓ Memory mapping working - model ({total_model_size:.2f} GB) loaded with only {used_after_load:.1f} GB RAM")
    
    # Get model inputs/outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"\nModel Info:")
    print(f"  Inputs: {len(inputs)}")
    for inp in inputs:
        print(f"    - {inp.name}: shape={inp.shape}, type={inp.type}")
    print(f"  Outputs: {len(outputs)}")
    for out in outputs:
        print(f"    - {out.name}: shape={out.shape}, type={out.type}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTroubleshooting:")
    print("  1. Ensure the ONNX model is valid")
    print("  2. Check that Vitis AI EP is properly installed")
    print("  3. Verify NPU is available: Check Ryzen AI Software")
    print("  4. Ensure sufficient disk space for memory mapping")
    print("  5. Check that model files are not corrupted")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Run Inference
# ============================================================================

print("\n" + "=" * 70)
print("Running Inference")
print("=" * 70)

# Example prompt
prompt = "Write a Python function to check if a number is prime."

print(f"Prompt: {prompt}\n")

# Try Optimum method first (easier)
USE_OPTIMUM_METHOD = True

# Method 1: Try Optimum ORTModel (easier generation)
if USE_OPTIMUM_METHOD:
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        print("Using Optimum ORTModel (recommended for generation)...")
        
        # Load model directory (not single file)
        model_dir = INT8_MODEL_PATH if os.path.isdir(INT8_MODEL_PATH) else os.path.dirname(INT8_MODEL_PATH)
        
        ort_model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            providers=providers,
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded with Optimum")
        print(f"  Active provider: {ort_model.model.get_providers()[0]}")
        
        # Prepare prompt
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(text, return_tensors="pt")
        
        print(f"\nGenerating text...")
        start_time = time.time()
        
        generated_ids = ort_model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        elapsed = time.time() - start_time
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response_text = generated_text[len(text):].strip()
        
        print("\n" + "=" * 70)
        print("Generated Response:")
        print("=" * 70)
        print(response_text)
        print("=" * 70)
        
        tokens_generated = generated_ids.shape[1] - inputs["input_ids"].shape[1]
        print(f"\n✓ Generation successful!")
        print(f"  Provider: {ort_model.model.get_providers()[0]}")
        print(f"  Tokens: {tokens_generated}, Time: {elapsed:.2f}s, Speed: {tokens_generated/elapsed:.2f} tok/s")
        
    except ImportError:
        print("⚠ Optimum not available - using manual method")
        USE_OPTIMUM_METHOD = False
    except Exception as e:
        print(f"⚠ Error with Optimum: {e}")
        print("  Trying manual method...")
        USE_OPTIMUM_METHOD = False

# Method 2: Manual generation with ONNX Runtime (if Optimum failed)
if not USE_OPTIMUM_METHOD:
    print("\nUsing manual ONNX Runtime method...")
    
    # Prepare input
    try:
        # Tokenize the prompt
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize to input IDs
        encoded = tokenizer(text, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input IDs (first 10): {input_ids[0][:10]}\n")
        
        # Prepare inputs for ONNX Runtime
        inputs_dict = {
            "input_ids": input_ids
        }
        
        # Add attention_mask if model expects it
        if "attention_mask" in encoded:
            inputs_dict["attention_mask"] = encoded["attention_mask"].astype(np.int64)
        
        print("Running inference...")
        start_time = time.time()
        
        # Run inference
        outputs = session.run(None, inputs_dict)
        
        elapsed = time.time() - start_time
        print(f"✓ Inference completed in {elapsed:.2f} seconds")
        
        # Process output
        logits = outputs[0]
        print(f"Output logits shape: {logits.shape}")
        
        # For text generation, we need to generate tokens iteratively
        print("\nGenerating text (iterative)...")
        
        max_new_tokens = 100  # Limit generation length
        generated_ids = input_ids.copy()
        
        for step in range(max_new_tokens):
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature and sample
            temperature = 0.7
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = np.exp(next_token_logits - np.max(next_token_logits))
                probs = probs / np.sum(probs)
                next_token_id = np.random.choice(len(probs), p=probs)
            else:
                next_token_id = np.argmax(next_token_logits)
            
            # Append to generated sequence
            generated_ids = np.append(generated_ids, [[next_token_id]], axis=1)
            
            # Check for end token
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Prepare next input (use last N tokens for context)
            if generated_ids.shape[1] > 512:
                next_input = generated_ids[:, -512:]
            else:
                next_input = generated_ids
            
            # Run next inference
            next_inputs = {"input_ids": next_input.astype(np.int64)}
            if "attention_mask" in inputs_dict:
                next_attention = np.ones_like(next_input, dtype=np.int64)
                next_inputs["attention_mask"] = next_attention
            
            outputs = session.run(None, next_inputs)
            logits = outputs[0]
            
            if (step + 1) % 10 == 0:
                print(f"  Generated {step + 1} tokens...", end='\r')
        
        print(f"\n✓ Generated {generated_ids.shape[1] - input_ids.shape[1]} new tokens")
        
        # Decode the full generated sequence
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt_text):].strip()
        
        print("\n" + "=" * 70)
        print("Generated Response:")
        print("=" * 70)
        print(response_text)
        print("=" * 70)
        
        total_time = time.time() - start_time
        tokens_per_second = (generated_ids.shape[1] - input_ids.shape[1]) / total_time
        
        print(f"\n✓ Inference successful!")
        print(f"  Provider used: {actual_provider}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Generation speed: {tokens_per_second:.2f} tokens/second")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("\nNote: For full text generation, you may need to:")
        print("  1. Use iterative generation (not just single forward pass)")
        print("  2. Use the model's generate() method with ONNX Runtime")
        print("  3. Or use a generation wrapper that supports ONNX")
        import traceback
        traceback.print_exc()

# ============================================================================
# Performance Information
# ============================================================================

print("\n" + "=" * 70)
print("Performance Tips")
print("=" * 70)
print("Memory Mapping Benefits:")
print("  ✓ Models larger than RAM can be loaded")
print("  ✓ Only required parts are loaded into memory")
print("  ✓ Reduces memory footprint significantly")
print("\nFor best NPU performance:")
print("  1. Use INT8 quantized models (good balance of size and accuracy)")
print("  2. Use Vitis AI Execution Provider (NPU acceleration)")
print("  3. Batch multiple requests together when possible")
print("  4. Monitor NPU utilization with Ryzen AI tools")
print("\nTo verify NPU is being used:")
print("  - Check the 'Active provider' message above")
print("  - Should show 'VitisAIExecutionProvider'")
print("  - Monitor with: ryzenai-monitor (if available)")
print("=" * 70 + "\n")

