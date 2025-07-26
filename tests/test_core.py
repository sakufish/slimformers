import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from peft import TaskType
from slimformers import Pruner
from slimformers import lora_finetune

# Load model and tokenizer
# model_id = "deepseek-ai/deepseek-coder-1.3b-base"
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inputs
text = "The quick brown fox:"
inputs = tokenizer(text, return_tensors="pt")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Prune MLP layers
pruner = Pruner(model)
pruner.prune_all_mlp_layers(inputs, sparsity=0.4)

print("After pruning:")
print(f"Pruned model size: {count_parameters(model):,} params")

# Forward pass and generation test
model.eval()
with torch.no_grad():
    out = model(**inputs)
print("Forward pass OK, logits.shape =", out.logits.shape)

gen_ids = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)
print("Generated (pruned) text:\n", tokenizer.decode(gen_ids[0], skip_special_tokens=True))

# Create a simple dataloader for LoRA
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.encodings.items()}

dataloader = DataLoader(TextDataset(inputs), batch_size=1)

# Apply LoRA 
print("\nStarting LoRA fine-tuning...")
model = lora_finetune(
    model=model,
    dataloader=dataloader,
    epochs=20,
    lr=1e-4,
    device="cpu",
    r=8,
    alpha=16,
    dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

print("\nAfter LoRA fine-tuning:")
print(f"Fine-tuned model size: {count_parameters(model):,} params")

# Test generation
model.eval()
with torch.no_grad():
    out_ft = model(**inputs)
print("Forward pass after LoRA, logits.shape =", out_ft.logits.shape)

gen_ids_ft = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)
print("Generated (LoRA-finetuned) text:\n", tokenizer.decode(gen_ids_ft[0], skip_special_tokens=True))

# Print stats
original_model = AutoModelForCausalLM.from_pretrained(model_id)
orig_params = count_parameters(original_model)
pruned_params = count_parameters(model)
print(f"\nOriginal size: {orig_params:,} params")
print(f"Pruned+LoRA size: {pruned_params:,} params")
print(f"Reduction: {orig_params - pruned_params:,} params ({100 * (orig_params - pruned_params) / orig_params:.2f}%)")