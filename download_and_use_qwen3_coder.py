"""
Script to download and use Qwen3-Coder-30B model
The model will be automatically downloaded from Hugging Face when you first run this.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Option 1: Automatic download (default behavior)
# The model will be downloaded to Hugging Face cache directory
# Windows: C:\Users\<username>\.cache\huggingface\hub\
# Linux: ~/.cache/huggingface/hub/

model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  # Full model name from Hugging Face

print(f"Loading model: {model_name}")
print("Note: First run will download ~60GB of model files to Hugging Face cache.")
print("This may take a while depending on your internet connection.\n")

# Load tokenizer and model
# This will automatically download if not already cached
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
    device_map="auto",  # Automatically uses available GPU/CPU
    trust_remote_code=True
)

print("\nModel loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}\n")

# Example usage
prompt = "Write a Python function to check if a number is prime."
print(f"Prompt: {prompt}\n")

# Format prompt for chat
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Tokenize and generate
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,  # Adjust based on your needs
    temperature=0.7,
    do_sample=True
)

# Decode output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("Generated response:")
print(content)
