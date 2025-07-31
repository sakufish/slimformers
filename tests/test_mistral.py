import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType
from slimformers import Pruner, lora_finetune

# Load Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Sample corpus
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "LoRA and pruning improve model efficiency.",
    "Transformers are powerful neural networks."
]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Wrap in Dataset + DataLoader
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings["input_ids"].size(0)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

dataloader = DataLoader(TextDataset(encodings), batch_size=2, shuffle=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Run pruning
pruner = Pruner(model)
pruner.prune_attention_heads(dataloader=dataloader, sparsity=0.4)
pruner.prune_all_mlp_layers(dataloader=dataloader, sparsity=0.4)

print("After pruning:")
print(f"Pruned model size: {count_parameters(model):,} params")

# Forward test
model.eval()
sample_inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cpu")
with torch.no_grad():
    out = model(**sample_inputs)
print("Forward pass OK, logits.shape =", out.logits.shape)

# Generation test
gen_ids = model.generate(
    **sample_inputs,
    max_new_tokens=20,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)
print("Generated (pruned) text:\n", tokenizer.decode(gen_ids[0], skip_special_tokens=True))

# LoRA fine-tuning
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

# Final test
model.eval()
with torch.no_grad():
    out_ft = model(**sample_inputs)
print("Forward pass after LoRA, logits.shape =", out_ft.logits.shape)

gen_ids_ft = model.generate(
    **sample_inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)
print("Generated (LoRA-finetuned) text:\n", tokenizer.decode(gen_ids_ft[0], skip_special_tokens=True))

# Print final report
pruner.report()