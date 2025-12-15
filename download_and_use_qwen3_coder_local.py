"""
Script to download Qwen3-Coder-30B model to a LOCAL folder
Use this if you want the model files in a specific directory (e.g., same folder as script)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Option 2: Download to a local folder
# Set this to where you want the model stored
MODEL_FOLDER = "./models/Qwen3-Coder-30B-A3B-Instruct-FP8"  # Local folder path
model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# Create directory if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

print(f"Downloading model to: {os.path.abspath(MODEL_FOLDER)}")
print("This will download ~60GB of model files.")
print("First download may take a while depending on your internet connection.\n")

# Download and save to local folder
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=None  # Don't use cache, download directly
)
tokenizer.save_pretrained(MODEL_FOLDER)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=None  # Don't use cache, download directly
)
model.save_pretrained(MODEL_FOLDER)

print(f"\nModel saved to: {os.path.abspath(MODEL_FOLDER)}")

# Now load from local folder
print("\nLoading model from local folder...")
local_tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER, trust_remote_code=True)
local_model = AutoModelForCausalLM.from_pretrained(
    MODEL_FOLDER,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded from local folder successfully!")
print(f"Model device: {next(local_model.parameters()).device}\n")

# Example usage
prompt = "Write a Python function to calculate fibonacci numbers."
print(f"Prompt: {prompt}\n")

messages = [
    {"role": "user", "content": prompt}
]
text = local_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = local_tokenizer([text], return_tensors="pt").to(local_model.device)

generated_ids = local_model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = local_tokenizer.decode(output_ids, skip_special_tokens=True)

print("Generated response:")
print(content)
