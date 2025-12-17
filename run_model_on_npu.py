"""
Script to run quantized ONNX model on AMD Ryzen AI NPU
Uses Vitis AI Execution Provider with ONNX Runtime

Supports:
- AMD Ryzen AI NPU (mobile/APU with NPU) - uses Vitis AI EP
- NVIDIA GPU - uses CUDA EP (fallback)
- CPU (any x86_64) - uses CPU EP with multi-threading (fallback)
  - Works on Ryzen 9950X, Intel CPUs, etc.
  - Automatically uses all available CPU threads

Prerequisites:
1. ONNX Runtime installed (required)
2. For NPU: AMD Ryzen AI Software with Vitis AI EP
3. Quantized ONNX model (INT4/BF16 recommended)

Usage:
    python run_model_on_npu.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Path to your quantized ONNX model
# Use the quantized model (INT4/BF16) for best NPU performance
QUANTIZED_MODEL_PATH = "./models/qwen3-coder-30b-onnx-int4/model_int4.onnx"

# Alternative: Use non-quantized ONNX model (may be slower/larger)
# QUANTIZED_MODEL_PATH = "./models/qwen3-coder-30b-onnx/model.onnx"

# Path to tokenizer (needed for text processing)
TOKENIZER_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct"

# ============================================================================
# Check Dependencies
# ============================================================================

print("=" * 70)
print("AMD Ryzen AI NPU Inference")
print("(Also works on CPU/GPU - automatic fallback)")
print("=" * 70)

# Check for ONNX Runtime
try:
    from onnxruntime import InferenceSession
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
# Load ONNX Model for NPU
# ============================================================================

print("\n" + "=" * 70)
print("Loading ONNX Model for NPU")
print("=" * 70)

# Find the ONNX model file
if os.path.isdir(QUANTIZED_MODEL_PATH):
    onnx_files = list(Path(QUANTIZED_MODEL_PATH).glob("*.onnx"))
    if not onnx_files:
        print(f"❌ No .onnx files found in {QUANTIZED_MODEL_PATH}")
        sys.exit(1)
    model_file = str(max(onnx_files, key=lambda p: p.stat().st_size))
    print(f"Found ONNX model: {model_file}")
else:
    model_file = QUANTIZED_MODEL_PATH
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("\nAvailable options:")
        print("  1. Run export_to_onnx_with_quark.py to create quantized model")
        print("  2. Run quantize_onnx_model.py to quantize existing ONNX model")
        print("  3. Update QUANTIZED_MODEL_PATH in this script")
        sys.exit(1)

print(f"Loading model: {model_file}")

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
    import onnxruntime
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

# Configure CPU session options for optimal performance
# This is especially important for high-end CPUs like Ryzen 9950X
import multiprocessing
cpu_threads = multiprocessing.cpu_count()

session_options = None
if not HAS_VITIS_AI or 'CPUExecutionProvider' in providers:
    # Configure CPU provider options for multi-threading
    try:
        import onnxruntime
        from onnxruntime import SessionOptions
        session_options = SessionOptions()
        
        # Use all available CPU threads
        session_options.intra_op_num_threads = cpu_threads
        session_options.inter_op_num_threads = cpu_threads
        
        # Enable optimizations
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization (reduces memory allocations)
        session_options.enable_mem_pattern = True
        
        # Enable memory arena (improves performance)
        session_options.enable_mem_arena = True
        
        print(f"✓ CPU optimizations configured:")
        print(f"  - Using {cpu_threads} threads (intra_op and inter_op)")
        print(f"  - Graph optimizations: Enabled")
        print(f"  - Memory optimizations: Enabled")
    except Exception as e:
        print(f"⚠ Could not configure CPU options: {e}")
        session_options = None

# Create inference session
try:
    print("\nCreating inference session...")
    session = InferenceSession(
        model_file,
        providers=providers,
        sess_options=session_options
    )
    
    # Check which provider is actually being used
    actual_provider = session.get_providers()[0]
    print(f"✓ Model loaded successfully!")
    print(f"  Active provider: {actual_provider}")
    
    if actual_provider == 'VitisAIExecutionProvider':
        print("  🎉 Running on AMD Ryzen AI NPU!")
    elif actual_provider == 'CUDAExecutionProvider':
        print("  ⚠ Running on GPU (NPU not available)")
    else:
        print("  ⚠ Running on CPU (NPU/GPU not available)")
        print(f"  💡 Using {cpu_threads} CPU threads for inference")
        print("  Note: CPU inference will be slower than NPU, but still functional")
        print("  For Ryzen 9950X: CPU execution is expected (no NPU in desktop CPUs)")
    
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
    print("  4. Try using CPU provider first to test the model")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Run Inference
# ============================================================================

print("\n" + "=" * 70)
print("Running Inference on NPU")
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
        model_dir = QUANTIZED_MODEL_PATH if os.path.isdir(QUANTIZED_MODEL_PATH) else os.path.dirname(QUANTIZED_MODEL_PATH)
        
        ort_model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            providers=providers,
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded")
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
        # ONNX models typically expect input_ids as numpy array
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
        # The output is typically logits (shape: [batch, sequence, vocab_size])
        logits = outputs[0]
        print(f"Output logits shape: {logits.shape}")
        
        # For text generation, we need to generate tokens iteratively
        # This is a simplified example - shows how to do basic generation
        print("\nGenerating text (iterative)...")
        
        max_new_tokens = 100  # Limit generation length
        generated_ids = input_ids.copy()
        
        for step in range(max_new_tokens):
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature and sample (or use greedy: argmax)
            temperature = 0.7
            if temperature > 0:
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                # Softmax
                probs = np.exp(next_token_logits - np.max(next_token_logits))
                probs = probs / np.sum(probs)
                # Sample
                next_token_id = np.random.choice(len(probs), p=probs)
            else:
                # Greedy decoding
                next_token_id = np.argmax(next_token_logits)
            
            # Append to generated sequence
            generated_ids = np.append(generated_ids, [[next_token_id]], axis=1)
            
            # Check for end token
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Prepare next input (use last N tokens for context, or full sequence if small)
            # For efficiency, you might want to use only the last tokens
            if generated_ids.shape[1] > 512:  # Limit context window
                next_input = generated_ids[:, -512:]
            else:
                next_input = generated_ids
            
            # Run next inference
            next_inputs = {"input_ids": next_input.astype(np.int64)}
            if "attention_mask" in inputs_dict:
                # Update attention mask
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
print("For best NPU performance:")
print("  1. Use INT4 quantized models (smallest, fastest)")
print("  2. Use BF16 for better accuracy if INT4 is too aggressive")
print("  3. Batch multiple requests together when possible")
print("  4. Monitor NPU utilization with Ryzen AI tools")
print("  5. For generation, use Optimum's ORTModel.generate() method")
print("\nFor CPU execution (e.g., Ryzen 9950X):")
print("  ✓ Script automatically uses all available CPU threads")
print("  ✓ Graph and memory optimizations are enabled")
print("  ⚠ CPU inference will be slower than NPU (but still functional)")
print("  ⚠ Expected speed: 1-10 tokens/second (depends on model size)")
print("  💡 For better CPU performance:")
print("     - Use INT4 quantized models (smaller, faster)")
print("     - Close other CPU-intensive applications")
print("     - Consider using a GPU if available")
print("\nTo verify execution provider:")
print("  - Check the 'Active provider' message above")
print("  - NPU: 'VitisAIExecutionProvider'")
print("  - GPU: 'CUDAExecutionProvider'")
print("  - CPU: 'CPUExecutionProvider' (with thread count)")

