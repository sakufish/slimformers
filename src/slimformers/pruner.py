import torch
from torch import nn
from transformers.modeling_utils import Conv1D
from rich.console import Console
from rich.panel import Panel
from .discovery import DISCOVERY_REGISTRY, default_discover, ATTENTION_DISCOVERY_REGISTRY
import psutil
import os

console = Console()

class Pruner:
    def __init__(self, model: nn.Module, pruning_strategy=None):
        """
        Set up the Pruner with a model and optional strategy.
        If no strategy is given, default to keeping neurons with the highest activation magnitudes.
        """
        self.model = model
        self.activations = {}
        self.pruning_strategy = pruning_strategy or self._compute_topk_neurons
        self.initial_params_num = sum(p.numel() for p in model.parameters())
        
        self._init_cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        if torch.cuda.is_available():
            device = next(model.parameters()).device
            torch.cuda.reset_peak_memory_stats(device)
            self._init_gpu_mem = torch.cuda.memory_allocated(device) / 1024**2
        else:
            self._init_gpu_mem = None
        
        console.rule("[bold cyan]Pruner Initialized")
        console.print(
            Panel.fit(
                f"Model: [bold]{type(model).__name__}[/bold]\n"
                f"Strategy: [bold]{self.pruning_strategy.__name__}[/bold]",
                title="Initialization Summary",
                border_style="cyan"
            )
        )

    @staticmethod
    def _discover_mlp_blocks(model: nn.Module):
        """
        Automatically find MLP/FFN blocks based on model type.
        Uses a registry of discovery functions (GPT2, BERT, LLaMA, etc).
        """
        cls = type(model).__name__
        finder = DISCOVERY_REGISTRY.get(cls, default_discover)
        return finder(model)

    def _compute_topk_neurons(self, activations: torch.Tensor, sparsity: float):
        """
        Select the top neurons (or heads) by activation magnitude.
        Supports:
        - 3D tensor: (batch, seq_len, features) → mean over (0,1)
        - 2D tensor: (batch, features)         → mean over 0
        - 1D tensor: (features,)               → use abs directly
        """
        if activations.dim() == 3:
            mags = activations.abs().mean(dim=(0, 1))
        elif activations.dim() == 2:
            mags = activations.abs().mean(dim=0)
        elif activations.dim() == 1:
            mags = activations.abs()
        else:
            raise ValueError(f"Bad activation shape {activations.shape}")

        total = mags.numel()
        k = int((1.0 - sparsity) * total)
        keep = torch.topk(mags, k=k).indices
        return keep, total


    def _hook_activations(self, layer_name: str):
        """
        Register a forward hook on a specific layer to capture its output.
        """
        module = dict(self.model.named_modules())[layer_name]
        return module.register_forward_hook(
            lambda mod, inp, out, key=layer_name: self.activations.setdefault(key, out.detach().cpu())
        )

    def _rebuild_linear(self, layer: nn.Linear, keep_out: torch.Tensor = None, keep_in:  torch.Tensor = None):
        """
        Rebuild a Linear layer by slicing its input/output weights.
        """
        W = layer.weight.data.cpu() 
        B = layer.bias.data.cpu() if layer.bias is not None else None

        if keep_out is not None and keep_in is None:
            new_W = W[keep_out, :]
        elif keep_out is None and keep_in is not None:
            new_W = W[:, keep_in]
        elif keep_out is not None and keep_in is not None:
            new_W = W[keep_out][:, keep_in]
        else:
            raise ValueError("Must provide keep_out and/or keep_in")

        out_f, in_f = new_W.shape
        new = nn.Linear(in_f, out_f, bias=(B is not None))
        new.weight.data = new_W.to(layer.weight.device)
        if B is not None:
            new.bias.data = (B[keep_out] if keep_out is not None else B).to(layer.bias.device)
        return new

    def _rebuild_conv1d(self, layer: Conv1D, keep_out: torch.Tensor = None, keep_in:  torch.Tensor = None):
        """
        Same as _rebuild_linear but for HuggingFace's Conv1D (used in GPT-2).
        """
        W = layer.weight.data.cpu()
        B = layer.bias.data.cpu() if layer.bias is not None else None

        if keep_out is not None and keep_in is None:
            new_W = W[:, keep_out]
        elif keep_out is None and keep_in is not None:
            new_W = W[keep_in, :]
        elif keep_out is not None and keep_in is not None:
            new_W = W[keep_in][:, keep_out]
        else:
            raise ValueError("Must provide keep_out and/or keep_in")

        in_c, out_c = new_W.shape
        new = Conv1D(out_c, in_c)
        new.weight.data = new_W.to(layer.weight.device)
        new.nf = out_c
        if B is not None:
            new.bias.data = (B[keep_out] if keep_out is not None else B).to(layer.bias.device)
        return new

    def _replace_module(self, name: str, new_mod: nn.Module):
        """
        Replaces a module in the model with the updated version.
        For example, replaces an old Linear with a sliced one.
        """
        parent_name, attr = name.rsplit('.', 1)
        parent = dict(self.model.named_modules())[parent_name]
        setattr(parent, attr, new_mod)

    def _rebuild(self, layer, keep_out=None, keep_in=None):
        """
        Calls the correct rebuild function depending on layer type.
        """
        if isinstance(layer, Conv1D):
            return self._rebuild_conv1d(layer, keep_out, keep_in)
        elif isinstance(layer, nn.Linear):
            return self._rebuild_linear(layer, keep_out, keep_in)
        else:
            raise TypeError(f"Can't rebuild module of type {type(layer)}")

    def prune_all_mlp_layers(self, dataloader: torch.utils.data.DataLoader, sparsity: float = 0.3, max_batches: int = 10):
        """
        Prune MLP/FFN layers using activations collected from multiple batches in a DataLoader.
        Automatically averages activations across batches before applying the pruning strategy.

        Args:
            dataloader (torch.utils.data.DataLoader): Yields dict[str, torch.Tensor] batches
            sparsity (float): Fraction of neurons to prune (0.0 keeps all, 1.0 prunes all)
            max_batches (int): Max number of batches to use for computing average activations
        """
        
        self.model.eval()
        device = next(self.model.parameters()).device

        console.rule(f"[bold]Starting Pruning at {sparsity:.0%} Sparsity")
        blocks = Pruner._discover_mlp_blocks(self.model)
        console.print(f"[bold]Discovered {len(blocks)} MLP blocks[/bold]\n")

        for i, blk in enumerate(blocks):
            console.print(f"[bold white]Block {i}[/bold white] – Type: {blk['type']}")

            hook_key = blk["gate_name"] if blk["type"] == "gated" else blk["fc_name"]
            
            self.activations = {}
            handle = self._hook_activations(hook_key)

            total_acts = None
            num_batches = 0

            for batch in dataloader:
                if num_batches >= max_batches:
                    break
                with torch.no_grad():
                    _ = self.model(**{k: v.to(device) for k, v in batch.items()})
                act = self.activations.pop(hook_key, None)
                if act is None:
                    continue
                total_acts = act if total_acts is None else total_acts + act
                num_batches += 1

            handle.remove()

            if num_batches == 0:
                raise RuntimeError(f"No activations captured for block {i} ({hook_key})")

            avg_acts = total_acts / num_batches
            keep_idx, orig = self.pruning_strategy(avg_acts, sparsity)

            if blk["type"] == "gated":
                new_gate = self._rebuild(blk["gate"],   keep_out=keep_idx)
                new_up   = self._rebuild(blk["up"],     keep_out=keep_idx)
                new_down = self._rebuild(blk["down"],   keep_in=keep_idx)

                self._replace_module(blk["gate_name"], new_gate)
                self._replace_module(blk["up_name"],   new_up)
                self._replace_module(blk["down_name"], new_down)

                console.print(f"  [green]Pruned gated MLP[/green]: {orig} → {keep_idx.numel()} units")

            else:
                new_fc   = self._rebuild(blk["fc"],   keep_out=keep_idx)
                new_proj = self._rebuild(blk["proj"], keep_in=keep_idx)

                self._replace_module(blk["fc_name"],   new_fc)
                self._replace_module(blk["proj_name"], new_proj)

                console.print(f"  [green]Pruned FFN[/green]: {orig} → {keep_idx.numel()} units")

        console.rule("[bold green]Pruning Complete")    

    def report(self):
        """
        Print the parameter savings after pruning.
        """
        current_params = sum(p.numel() for p in self.model.parameters())
        saved = self.initial_params_num - current_params
        percent = 100 * saved / self.initial_params_num

        proc = psutil.Process(os.getpid())
        final_cpu = proc.memory_info().rss / 1024**2
        cpu_diff = final_cpu - self._init_cpu_mem

        if torch.cuda.is_available():
            device = next(self.model.parameters()).device
            final_gpu = torch.cuda.memory_allocated(device) / 1024**2
            peak_gpu  = torch.cuda.max_memory_allocated(device) / 1024**2
            gpu_diff = final_gpu - self._init_gpu_mem
            gpu_line = (
                f"[bold]GPU Memory (Before --> After):[/bold] "
                f"{self._init_gpu_mem:.2f} MB --> {final_gpu:.2f} MB "
                f"([bold green]{gpu_diff:+.2f} MB[/bold green])\n"
                f"[bold]GPU Memory (Peak):[/bold] {peak_gpu:.2f} MB"
            )
        else:
            gpu_line = "[dim]GPU not available — skipped[/dim]"

        cpu_line = (
            f"[bold]CPU Memory (Before --> After):[/bold] "
            f"{self._init_cpu_mem:.2f} MB --> {final_cpu:.2f} MB "
            f"([bold green]{cpu_diff:+.2f} MB[/bold green])"
        )

        console.rule("[bold magenta]Pruning Summary")
        console.print(
            Panel.fit(
                f"[bold]Original Parameters:[/bold] {self.initial_params_num:,}\n"
                f"[bold]Pruned Parameters:[/bold] {current_params:,}\n"
                f"[bold green]Total Reduction:[/bold green] {saved:,} ({percent:.2f}%)\n\n"
                f"{gpu_line}\n{cpu_line}",
                title="[bold]Compression Results[/bold]",
                border_style="magenta"
            )
        )

    def prune_attention_heads(self, dataloader, sparsity=0.3, max_batches=10):
        """
        Prune attention heads using average query (or packed QKV) activations.
        So far, it supports GPT‑2 (packed) and BERT/LLaMA (separate).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        console.rule(f"[bold]Starting Attention Pruning at {sparsity:.0%} Sparsity")

        blocks = ATTENTION_DISCOVERY_REGISTRY.get(type(self.model).__name__, lambda m: [])(self.model)
        console.print(f"[bold]Discovered {len(blocks)} attention blocks[/bold]\n")

        for blk in blocks:
            prefix = blk["prefix"]
            key = blk.get("qkv_name") or blk["q_name"]
            handle = self._hook_activations(key)
            total_acts, n = None, 0
            for batch in dataloader:
                if n >= max_batches:
                    break
                with torch.no_grad():
                    _ = self.model(**{k: v.to(device) for k, v in batch.items()})
                act = self.activations.pop(key, None)
                if act is not None:
                    total_acts = act if total_acts is None else total_acts + act
                    n += 1
            handle.remove()
            if n == 0:
                raise RuntimeError(f"No activations for attention block {prefix}")

            avg_act = (total_acts / n).abs()    
            B, T, D = avg_act.shape
            H = blk["num_heads"]
            head_dim_qkv = D // H

            head_mags = avg_act.view(B, T, H, head_dim_qkv).mean(dim=(0, 1, 3))
            keep, orig = self.pruning_strategy(head_mags, sparsity)

            idx_qkv = torch.cat([
                torch.arange(h * head_dim_qkv, (h + 1) * head_dim_qkv, device=keep.device)
                for h in keep
            ])

            out_layer = blk["out"]
            if isinstance(out_layer, Conv1D):
                in_dim = out_layer.weight.size(0)
            else:
                in_dim = out_layer.in_features
            head_dim_out = in_dim // H
            idx_out = torch.cat([
                torch.arange(h * head_dim_out, (h + 1) * head_dim_out, device=keep.device)
                for h in keep
            ])

            if blk["type"] == "packed":
                new_qkv = self._rebuild(blk["qkv"], keep_out=idx_qkv)
                self._replace_module(blk["qkv_name"], new_qkv)
            else:
                for name in ("q", "k", "v"):
                    new_lin = self._rebuild(blk[name], keep_out=idx_qkv)
                    self._replace_module(blk[f"{name}_name"], new_lin)

            new_out = self._rebuild(blk["out"], keep_in=idx_out)
            self._replace_module(blk["out_name"], new_out)
            self._patch_attention_module(prefix, keep.numel(), head_dim_out)

            console.print(f"[green]Pruned {prefix}[/green]: {orig} → {keep.numel()} heads")

        console.rule("[bold green]Attention Pruning Complete")

    def _patch_attention_module(self, prefix: str, new_heads: int, head_dim_out: int):
        """
        Dynamically update any head-count or size attributes on the attn module.
        """
        mod = dict(self.model.named_modules())[prefix]

        if hasattr(mod, "num_heads"):
            mod.num_heads = new_heads
        if hasattr(mod, "num_attention_heads"):
            mod.num_attention_heads = new_heads

        if hasattr(mod, "embed_dim") and hasattr(mod, "split_size"):
            new_embed = head_dim_out * new_heads
            mod.embed_dim   = new_embed
            mod.split_size  = new_embed

        if hasattr(mod, "attention_head_size") and hasattr(mod, "all_head_size"):
            mod.all_head_size = mod.attention_head_size * new_heads


    def _collect_activations(self, blk, dataloader, max_batches):
        """
        Runs up to max_batches through the model, hooks either Q or packed QKV,
        and returns the avg activation tensor (B, T, D).
        """
        key = blk.get("qkv_name") or blk["q_name"]
        handle = self._hook_activations(key)
        total, n = None, 0
        device = next(self.model.parameters()).device

        for batch in dataloader:
            if n >= max_batches: break
            with torch.no_grad():
                _ = self.model(**{k: v.to(device) for k, v in batch.items()})
            act = self.activations.pop(key, None)
            if act is not None:
                total = act if total is None else total + act
                n += 1

        handle.remove()
        if n == 0:
            raise RuntimeError(f"No activations for {blk['prefix']}")
        return total / n