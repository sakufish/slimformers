import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import TaskType
from slimformers import Pruner
from slimformers import lora_finetune

# Load model and tokenizer
model_id = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Pad token
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# Sample corpus
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "LoRA and pruning improve model efficiency.",
    "Transformers are powerful neural networks."
]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Wrap tokenized data in a Dataset for DataLoader
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

# Prune MLP layers
pruner = Pruner(model)
pruner.prune_attention_heads(dataloader=dataloader, sparsity=0.4)
pruner.prune_all_mlp_layers(dataloader=dataloader, sparsity=0.4)

print("After pruning:")
print(f"Pruned model size: {count_parameters(model):,} params")

# Forward pass test
model.eval()
sample_inputs = tokenizer("The quick brown [MASK] jumps over the lazy dog.", return_tensors="pt")
with torch.no_grad():
    out = model(**sample_inputs)
print("Forward pass OK, logits.shape =", out.logits.shape)

# Decode prediction for [MASK]
mask_index = (sample_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
pred_token_id = out.logits[0, mask_index].argmax(dim=-1)
predicted_token = tokenizer.decode(pred_token_id)
print("Predicted token for [MASK]:", predicted_token)

# Apply LoRA
print("\nStarting LoRA fine-tuning...")
model = lora_finetune(
    model=model,
    dataloader=dataloader,
    epochs=5,
    lr=1e-4,
    device="cpu",
    r=8,
    alpha=16,
    dropout=0.05,
    task_type=TaskType.TOKEN_CLS,
)

print("\nAfter LoRA fine-tuning:")
print(f"Fine-tuned model size: {count_parameters(model):,} params")

# Test forward pass again
model.eval()
with torch.no_grad():
    out_ft = model(**sample_inputs)
print("Forward pass after LoRA, logits.shape =", out_ft.logits.shape)

# Decode new prediction
pred_token_id_ft = out_ft.logits[0, mask_index].argmax(dim=-1)
predicted_token_ft = tokenizer.decode(pred_token_id_ft)
print("Predicted token after LoRA:", predicted_token_ft)

# Stats
pruner.report()