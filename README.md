# Slimformers

Slimformers is a lightweight Python framework for pruning and fine-tuning transformer models. It supports activation-based MLP (FFN) pruning, attention head pruning, low-rank adaptation (LoRA) without needing any manual layer specification.

# Features

- Prunes neurons based on average activations across multiple batches
- Prunes attention heads based on mean query activations
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
## Prune Attention Heads
``` python
# Prune 40% of attention heads based on query activations
pruner.prune_attention_heads(
    dataloader=dataloader,
    sparsity=0.4,
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
## Pruning Report

After pruning, ```pruner.report()``` displays a summary of the compression results. This includes:
- Original and pruned parameters counts
- Percentage reduction model size
- CPU and GPU memory usage before and after pruning
- Peak GPU memory usage (if CUDA enabled)

### Example 

Pruning was run on ```deepseek-ai/deepseek-coder-1.3b-base``` with 40% sparsity using a Lenovo ThinkPad T490 (Intel i5-8365U CPU, no GPU!): 
- Original Parameters: ```1,346,471,936```
- Pruned Parameters: ```1,024,855,424```
- Total Reduction: ```321,616,512 (23.89%)```
- CPU Memory: ```(Before --> After): 5398.57 MB --> 4253.34 MB (–1145.23 MB)```

# Limitations

Slimformers is made to be lightweight and architecture agnostic, but there are current limitations:

- **Limited model support (for now)**  
  Currently, attention head and FFN pruning supports GPT‑2, BERT, and LLaMA type models. Encoder-decoder architectures like T5 or BART (with cross-attention), and other variants like Falcon or BLOOM, are not supported yet. Also, FFN pruning assumes standard `nn.Linear` or `Conv1D` layers. If your model uses custom MLP designs like SwiGLU, Gated FFNs, or fused blocks, you'll need to add custom discovery logic.

  That said, **support for more models will be added over time**. The framework is modular, and the discovery system is easy to extend. Feel free to contribute or fork it to add support for other architectures. I will continue to expand the library's coverage.

- **Won’t work with exotic attention layouts**  
  If your model uses grouped heads, custom fused QKV projections, or MoE-style head routing, the default slicing logic might fail. This is rare for most Hugging Face models, but possible.

- **Not optimized for speed (Yet!)** 