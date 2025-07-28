# Slimformers

Slimformers is a lightweight Python framework for pruning and fine-tuning transformer models. It supports activation-based MLP (FFN) pruning and low-rank adaptation (LoRA) without the need for any manual layer specification.

# Features

- Prunes neurons based on average activations across multiple batches
- Automatic FFN and gated FFN block discovery for common architectures (GPT-2, BERT, LLaMA)
- Safely rebuilds pruned `nn.Linear` and `Conv1D` layers
- LoRA fine-tuning with auto-inferred target modules
- Compatible with Hugging Face models and tokenizers

# Quick Start

## Basic Pruning

```python
from slimformers import Pruner
from transformers import AutoModel, AutoTokenizer
import torch

# Load your model
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create pruner
pruner = Pruner(model)

# Prepare your data (returns dict with input_ids, attention_mask, etc.)
dataloader = your_dataloader_here

# Prune 30% of neurons based on activation magnitudes
pruner.prune_all_mlp_layers(
    dataloader=dataloader,
    sparsity=0.3,
    max_batches=10
)
```
## LoRA Fine-tuning
``` python
from slimformers import lora_finetune
from peft import TaskType

# Fine-tune with LoRA after pruning
fine_tuned_model = lora_finetune(
    model=model,
    dataloader=train_dataloader,
    epochs=3,
    lr=1e-4,
    device="cuda",
    r=8,
    alpha=16,
    task_type=TaskType.TOKEN_CLS
)
```
## Custom Prune Strategy
``` python
def custom_neuron_selection(activations, sparsity):
    """Custom strategy: keep neurons with highest variance"""
    if activations.dim() == 3:
        variance = activations.var(dim=(0,1))
    else:
        variance = activations.var(dim=0)
    
    total = variance.size(0)
    k = int((1.0 - sparsity) * total)
    return torch.topk(variance, k=k).indices, total

# Use custom strategy
pruner = Pruner(model, pruning_strategy=custom_neuron_selection)
```