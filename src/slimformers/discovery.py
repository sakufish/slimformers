from torch import nn
from transformers.modeling_utils import Conv1D

def discover_gpt2_ffns(model):
    """
    Locate MLP blocks in GPT-2 style models using c_fc and c_proj.
    """
    blocks = []
    for i, block in enumerate(model.transformer.h):
        blocks.append({
            "type": "ffn",
            "fc_name":   f"transformer.h.{i}.mlp.c_fc",
            "proj_name": f"transformer.h.{i}.mlp.c_proj",
            "fc":        block.mlp.c_fc,
            "proj":      block.mlp.c_proj,
        })
    return blocks

def discover_bert_ffns(model):
    """
    Locate FFN blocks in BERT models.
    """
    core = getattr(model, "bert", model)
    blocks = []

    for i, layer in enumerate(core.encoder.layer):
        blocks.append({
            "type": "ffn",
            "fc_name":   f"bert.encoder.layer.{i}.intermediate.dense",
            "proj_name": f"bert.encoder.layer.{i}.output.dense",
            "fc":        layer.intermediate.dense,
            "proj":      layer.output.dense,
        })
    return blocks

def discover_llama_ffns(model):
    """
    Locate gated FFN blocks in LLaMA models using
    gate_proj, up_proj, and down_proj.
    """
    blocks = []
    core = getattr(model, "model", model)

    for i, layer in enumerate(core.layers):
        mlp = layer.mlp
        blocks.append({
            "type": "gated",
            "gate_name": f"model.layers.{i}.mlp.gate_proj",
            "up_name":   f"model.layers.{i}.mlp.up_proj",
            "down_name": f"model.layers.{i}.mlp.down_proj",
            "gate":      mlp.gate_proj,
            "up":        mlp.up_proj,
            "down":      mlp.down_proj,
        })
    return blocks

# Registry mapping HuggingFace model class names to discovery functions
DISCOVERY_REGISTRY = {
    "GPT2Model":           discover_gpt2_ffns,
    "GPT2LMHeadModel":     discover_gpt2_ffns,
    "BertModel":           discover_bert_ffns,
    "BertForMaskedLM":     discover_bert_ffns,
    "LlamaModel":          discover_llama_ffns,
    "LlamaForCausalLM":    discover_llama_ffns,
}

def discover_ffns_model_agnostic(model, min_hidden_dim=128):
    """
    Generic fallback - scan all named modules, group by parent path,
    and infer FFN or gated FFN blocks from layer patterns.
    """
    all_mods = dict(model.named_modules())
    blocks = []

    # Group Linear / Conv1D layers by shared prefix
    grouped = {}
    for name, mod in all_mods.items():
        if isinstance(mod, (nn.Linear, Conv1D)):
            prefix = ".".join(name.split(".")[:-1])
            grouped.setdefault(prefix, []).append((name, mod))

    for prefix, layers in grouped.items():
        layer_names = [n for n, _ in layers]
        suffixes = [n.rsplit(".", 1)[-1] for n in layer_names]

        # Handle gated FFN case
        if {"gate_proj", "up_proj", "down_proj"}.issubset(set(suffixes)):
            blocks.append({
                "type": "gated",
                "gate_name": f"{prefix}.gate_proj",
                "up_name":   f"{prefix}.up_proj",
                "down_name": f"{prefix}.down_proj",
                "gate":      all_mods[f"{prefix}.gate_proj"],
                "up":        all_mods[f"{prefix}.up_proj"],
                "down":      all_mods[f"{prefix}.down_proj"],
            })
            continue

        # Handle standard FFN case
        candidates = []
        for name, mod in layers:
            w0, w1 = mod.weight.shape
            if w0 != w1 and max(w0, w1) >= min_hidden_dim:
                candidates.append((name, mod.weight.numel()))

        if len(candidates) >= 2:
            candidates.sort(key=lambda x: x[1], reverse=True)
            fc_name, _   = candidates[0]
            proj_name, _ = candidates[1]

            blocks.append({
                "type": "ffn",
                "fc_name": fc_name,
                "proj_name": proj_name,
                "fc":  all_mods[fc_name],
                "proj": all_mods[proj_name],
            })

    return blocks

def default_discover(model):
    """
    Default discovery entry point when no specific handler exists.
    """
    return discover_ffns_model_agnostic(model)

def discover_gpt2_attention(model):
    blocks = []
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        blocks.append({
            "type": "packed",
            "prefix": f"transformer.h.{i}.attn",
            "qkv_name": f"transformer.h.{i}.attn.c_attn",
            "out_name": f"transformer.h.{i}.attn.c_proj",
            "qkv": attn.c_attn,
            "out": attn.c_proj,
            "num_heads": attn.num_heads,
        })
    return blocks

def discover_bert_attention(model):
    """
    Locate separate‐QKV attention blocks in BERT models.
    """
    blocks = []
    core = getattr(model, "bert", model)

    for i, layer in enumerate(core.encoder.layer):
        sa = layer.attention.self
        out_proj = layer.attention.output.dense

        blocks.append({
            "type": "separate",
            "prefix":   f"bert.encoder.layer.{i}.attention.self",
            "q_name":   f"bert.encoder.layer.{i}.attention.self.query",
            "k_name":   f"bert.encoder.layer.{i}.attention.self.key",
            "v_name":   f"bert.encoder.layer.{i}.attention.self.value",
            "out_name": f"bert.encoder.layer.{i}.attention.output.dense",
            "q":        sa.query,
            "k":        sa.key,
            "v":        sa.value,
            "out":      out_proj,
            "num_heads": sa.num_attention_heads,
        })
    return blocks


def discover_llama_attention(model):
    """
    Locate separate‐QKV attention blocks in LLaMA models.
    """
    from torch import nn
    from transformers.modeling_utils import Conv1D

    blocks = []
    core = getattr(model, "model", model)

    for i, layer in enumerate(core.layers):
        sa = layer.self_attn

        if hasattr(sa, "num_attention_heads"):
            num_heads = sa.num_attention_heads
        elif hasattr(sa, "num_heads"):
            num_heads = sa.num_heads
        elif hasattr(sa, "config") and hasattr(sa.config, "num_attention_heads"):
            num_heads = sa.config.num_attention_heads
        else:
            raise AttributeError(
                f"LlamaAttention at layer {i} has no num_attention_heads, "
                "num_heads, or config.num_attention_heads"
            )

        blocks.append({
            "type": "separate",
            "prefix":   f"model.layers.{i}.self_attn",
            "q_name":   f"model.layers.{i}.self_attn.q_proj",
            "k_name":   f"model.layers.{i}.self_attn.k_proj",
            "v_name":   f"model.layers.{i}.self_attn.v_proj",
            "out_name": f"model.layers.{i}.self_attn.o_proj",
            "q":        sa.q_proj,
            "k":        sa.k_proj,
            "v":        sa.v_proj,
            "out":      sa.o_proj,
            "num_heads": num_heads,
        })
    return blocks

ATTENTION_DISCOVERY_REGISTRY = {
    "GPT2Model": discover_gpt2_attention,
    "GPT2LMHeadModel": discover_gpt2_attention,
    "BertModel": discover_bert_attention,
    "BertForMaskedLM": discover_bert_attention,
    "LlamaModel": discover_llama_attention,
    "LlamaForCausalLM": discover_llama_attention,
}