# AMD Ryzen AI Qwen3-Coder-30B Manual

## Quick setup (start here)
```bash
conda create -n ryzenai python=3.10 -y
conda activate ryzenai
```

Install core packages (use the PyTorch index that matches your GPU/CPU stack):
```bash
pip install --upgrade pip
pip install -r requirements.txt
# If you need a specific wheel: pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
# Optional extras (install only if supported): onnxruntime-gpu, onnxruntime-genai, intel-extension-for-pytorch
```
If using Intel GPU:
https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.8.10%2Bxpu&os=windows&package=pip

Hardware and disk reminders:
- Qwen3-Coder-30B needs ~60GB disk for the full model; 4-bit variants ~20GB.
- 32GB RAM minimum; 64GB+ recommended. NPU runs favor smaller (≤7B) models.
- First download can take time; keep the terminal open until it finishes.

## Repository layout
- `download_and_use_qwen3_coder.py` — auto-downloads from Hugging Face cache and runs a sample prompt.
- `download_and_use_qwen3_coder_local.py` — downloads the full model into `./models/Qwen3-Coder-30B-A3B-Instruct/`.
- `load_model_from_local_folder.py` — loads an existing local model folder; includes ONNX export helpers for NPU.
- `RYZEN_AI_QWEN3_CODER_30B_GUIDE.md` — deeper background and options (ONNX, hybrid execution, Lemonade SDK).

## Option A: Automatic download & run (simplest)
```bash
python download_and_use_qwen3_coder.py
```
- Downloads to the Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub\` on Windows).
- Uses `device_map="auto"` so GPU/CPU is chosen automatically.
- Prints a sample completion for the built-in prompt.

## Option B: Download to a specific local folder
```bash
python download_and_use_qwen3_coder_local.py
```
- Saves into `./models/Qwen3-Coder-30B-A3B-Instruct/` (edit `MODEL_FOLDER` in the script if you want another path).
- After download, the script reloads from that folder and runs a sample prompt.
- Good when you want the model alongside your project instead of the global cache.

## Option C: Load an existing local model folder
1. Set `MODEL_PATH` in `load_model_from_local_folder.py` to your folder (e.g., `./models/Qwen3-Coder-30B-A3B-Instruct-int4-AutoRound`).
2. Run:
   ```bash
   python load_model_from_local_folder.py
   ```
- Script auto-detects NVIDIA GPU, Intel XPU (if `intel-extension-for-pytorch` is installed), or falls back to CPU.
- Uses `device_map="auto"` or explicit CPU/XPU mapping to avoid accidental offloading.

## Export to ONNX for Ryzen AI NPU (summary)
`load_model_from_local_folder.py` already contains the ONNX export flow using Optimum. To use it:
1. Ensure `optimum`, `onnx`, and `onnxruntime` are installed in the `ryzenai` env.
2. Keep `MODEL_PATH` pointing to your local model folder (quantized or not).
3. Run the script; the export section is active and will write ONNX files to `./models/qwen3-coder-30b-onnx/`.
4. For NPU/Vitis AI EP, quantize the ONNX output (INT4 or BF16) with AMD Quark, then create an ORT session with `VitisAIExecutionProvider` (see comments in the script and `RYZEN_AI_QWEN3_CODER_30B_GUIDE.md`).

Notes:
- Export of a 30B model is lengthy (hours) and memory-intensive; GPUs or Intel XPU reduce time significantly.
- If exporting a quantized (int4) model, expect precision warnings; re-quantizing the ONNX with AMD Quark is recommended.

## Tips and troubleshooting
- `huggingface-cli login` first if downloads are throttled or require auth.
- If you hit CUDA out-of-memory, try smaller `max_new_tokens`, or use 4-bit weights.
- For Windows, long paths may fail; keep the repo path short (e.g., `D:\projects\amd-npu-converting`).
- For purely CPU runs, keep expectations modest—generation will be slow for 30B.
- See `RYZEN_AI_QWEN3_CODER_30B_GUIDE.md` for hybrid NPU/GPU setups, validated smaller models, and Lemonade SDK usage.


