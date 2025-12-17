# Running Qwen3-Coder 30B with Ryzen AI Software on NPU

## Important Limitations

**Note:** AMD Ryzen AI Software has been validated with smaller Qwen models (1.5B, 3B, 7B), but **Qwen3-Coder 30B (30 billion parameters) is significantly larger** and may not run efficiently or at all directly on the NPU due to memory constraints. The NPU typically handles smaller models better.

## Option 1: Convert to ONNX and Deploy on NPU (If Supported)

### Prerequisites

1. **Install Ryzen AI Software:**
   - Download from [AMD Ryzen AI Documentation](https://ryzenai.docs.amd.com)
   - Install the complete software suite including:
     - Ryzen AI Software
     - ONNX Runtime with Vitis AI Execution Provider
     - Lemonade SDK (for LLM deployment)

2. **Verify Installation:**
   ```bash
   conda activate <ryzen_ai_env>
   cd %RYZEN_AI_INSTALLATION_PATH%/quicktest
   python quicktest.py
   ```

### Steps to Convert and Deploy

1. **Convert Qwen3-Coder-30B to ONNX Format:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch
   import onnx
   from onnxruntime.transformers.models.gpt2.convert_to_onnx import convert_to_onnx
   
   model_name = "Qwen/Qwen3-Coder-30B"
   
   # Load the model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.bfloat16,
       trust_remote_code=True
   )
   
   # Export to ONNX (this may require significant memory and time)
   # Note: You may need to export in chunks due to model size
   ```

2. **Quantize the Model (Recommended for NPU):**
   - Use AMD Quark Model Optimization Library to quantize to INT4 or BF16
   - BF16 is supported for transformer models on Ryzen AI NPU
   - INT4 quantization can reduce memory requirements significantly

3. **Compile for NPU:**
   ```python
   from onnxruntime import InferenceSession
   from onnxruntime.vitisai import VitisAIExecutionProvider
   
   # Create inference session with Vitis AI EP
   session = InferenceSession(
       "qwen3-coder-30b-quantized.onnx",
       providers=['VitisAIExecutionProvider']
   )
   ```

4. **Run Inference:**
   ```python
   # Prepare input
   prompt = "Write a Python function to calculate fibonacci numbers"
   inputs = tokenizer(prompt, return_tensors="np")
   
   # Run on NPU
   outputs = session.run(None, dict(inputs))
   ```

## Option 2: Hybrid Execution (NPU + GPU/CPU)

If the full 30B model doesn't fit on NPU, Ryzen AI supports hybrid execution:

```python
from onnxruntime import InferenceSession
from onnxruntime.vitisai import VitisAIExecutionProvider

# Hybrid mode: NPU for supported ops, GPU/CPU for others
session = InferenceSession(
    "qwen3-coder-30b.onnx",
    providers=['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

## Option 3: Use Quantized Version on CPU/GPU (Recommended for 30B)

Since 30B is very large, consider using a quantized version:

1. **Download Quantized Model:**
   - Use 4-bit or 8-bit quantized versions from Hugging Face
   - 4-bit quantized Qwen3-Coder-30B requires ~17.2GB RAM

2. **Run with Transformers:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch
   
   model_name = "Qwen/Qwen3-Coder-30B-Instruct"  # or quantized variant
   
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       device_map="auto",  # Automatically uses available hardware
       torch_dtype=torch.bfloat16,
       trust_remote_code=True,
       # For quantization:
       # load_in_4bit=True,  # or load_in_8bit=True
   )
   
   prompt = "Write a Python function to check if a number is prime."
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   outputs = model.generate(**inputs, max_new_tokens=200)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

## Option 4: Use Lemonade SDK (If Model is Supported)

AMD's Lemonade SDK provides easier deployment:

1. **Install Lemonade SDK:**
   ```bash
   pip install lemonade-ai
   ```

2. **Use Lemonade Server:**
   ```python
   from lemonade import LemonadeClient
   
   client = LemonadeClient()
   response = client.generate(
       prompt="Write a Python function...",
       model="qwen3-coder-30b"  # If available in Lemonade
   )
   ```

## System Requirements

- **Memory:** At least 32GB RAM (64GB+ recommended for 30B model)
- **Storage:** ~60GB+ for full model, ~20GB for quantized
- **NPU:** Compatible Ryzen AI processor (e.g., Ryzen 8040/8050 series)
- **OS:** Windows 11 or Linux with Ryzen AI support

## Troubleshooting

1. **Model Too Large for NPU:**
   - Use quantized versions (INT4/INT8)
   - Consider smaller Qwen models (7B) that are validated
   - Use hybrid execution mode

2. **ONNX Conversion Issues:**
   - Ensure ONNX opset version 17
   - May need to convert in chunks for 30B model
   - Use `onnxruntime.transformers` utilities

3. **Memory Errors:**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable quantization

4. **Performance:**
   - Smaller models (1.5B-7B) perform better on NPU
   - 30B model may be better suited for GPU/CPU hybrid execution
   - Monitor NPU utilization with Ryzen AI monitoring tools

5. **"No space left on device" Error (Linux/Ubuntu):**
   - This often occurs when `/tmp` (tmpfs) is full during ONNX export
   - Check space: `df -h /tmp`
   - **Solution 1 (Recommended):** The script automatically uses a custom TMPDIR to avoid this
   - **Solution 2:** Temporarily increase `/tmp` size:
     ```bash
     sudo mount -o remount,size=500G /tmp
     ```
   - **Solution 3:** Permanently increase `/tmp` size by editing `/etc/fstab`:
     ```bash
     # Add or modify this line (requires root):
     tmpfs /tmp tmpfs defaults,size=500G 0 0
     # Then remount:
     sudo mount -o remount /tmp
     ```
   - **Solution 4:** Use the script's automatic TMPDIR redirection (already implemented)

6. **Process Killed / "Killed" Message (Linux/Ubuntu):**
   - This usually means the process was killed by the OOM (Out of Memory) killer
   - Check system logs: `dmesg | grep -i "out of memory"` or `journalctl -k | grep -i oom`
   - **Solutions:**
     - **Increase swap space:**
       ```bash
       # Create 64GB swap file
       sudo fallocate -l 64G /swapfile
       sudo chmod 600 /swapfile
       sudo mkswap /swapfile
       sudo swapon /swapfile
       # Make permanent: Add to /etc/fstab
       echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
       ```
     - **Close other memory-intensive applications**
     - **Run with nohup to survive terminal disconnection:**
       ```bash
       nohup python load_model_from_local_folder_ubuntu.py > export.log 2>&1 &
       ```
     - **Use systemd-run with memory protection:**
       ```bash
       systemd-run --scope -p MemoryLimit=64G python load_model_from_local_folder_ubuntu.py
       ```
     - **Adjust OOM score (requires root):**
       ```bash
       # After starting the process
       echo -500 | sudo tee /proc/$(pgrep -f load_model_from_local_folder)/oom_score_adj
       ```
   - The script now includes automatic OOM protection and memory monitoring

## Alternative: Use Validated Smaller Models

If NPU performance is critical, consider using validated Qwen models:
- Qwen2-7B (validated for Ryzen AI)
- Qwen2.5-7B-Instruct (validated for Ryzen AI)

These are more likely to run efficiently on the NPU.

## Where to Get the Model

### Option 1: Automatic Download (Recommended)

The model is available on **Hugging Face** and will be automatically downloaded when you first use it:

- **Model Repository:** [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- **Full Model Name:** `Qwen/Qwen3-Coder-30B-A3B-Instruct`

When you run code with `from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")`, the transformers library will:
- Automatically download the model files (~60GB)
- Cache them in the Hugging Face cache directory:
  - **Windows:** `C:\Users\<username>\.cache\huggingface\hub\`
  - **Linux/Mac:** `~/.cache/huggingface/hub/`

**You don't need the model in the same folder as your scripts** - it's cached globally.

### Option 2: Download to Local Folder

If you want the model in a specific folder (e.g., same directory as your scripts):

1. **Use the provided script:** `download_and_use_qwen3_coder_local.py`
   - This downloads the model to `./models/Qwen3-Coder-30B-A3B-Instruct/`
   - You can modify the `MODEL_FOLDER` variable to change the location

2. **Then load from local folder:**
   ```python
   MODEL_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct"
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ...)
   ```

### Option 3: Manual Download from Hugging Face

1. Visit: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
2. Click "Files and versions" tab
3. Download all files (requires Hugging Face account and Git LFS)
4. Extract to your desired folder

### Model Files Structure

The model folder should contain:
```
Qwen3-Coder-30B-A3B-Instruct/
├── config.json
├── generation_config.json
├── model-*.safetensors (multiple files, ~60GB total)
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### Scripts Provided

- `download_and_use_qwen3_coder.py` - Auto-downloads and uses model (uses cache)
- `download_and_use_qwen3_coder_local.py` - Downloads to local folder
- `load_model_from_local_folder.py` - Loads from local folder

## Resources

- [AMD Ryzen AI Documentation](https://ryzenai.docs.amd.com)
- [ONNX Runtime Vitis AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [Qwen Models on Hugging Face](https://huggingface.co/Qwen)
- [Qwen3-Coder-30B-A3B-Instruct Model Page](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [Lemonade SDK Documentation](https://ryzenai.docs.amd.com/en/1.4/lemonade.html)
