import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

def lora_finetune(
    model,
    dataloader,
    epochs=1,
    lr=1e-4,
    device="cpu",
    r=8,
    alpha=16,
    dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
):
    """
    Fine-tunes a transformer model using LoRA (Low-Rank Adaptation).
    
    Args:
        model (nn.Module): Pretrained transformer model
        dataloader (DataLoader): DataLoader providing training batches
        epochs (int): Number of training epochs
        lr (float): Learning rate for the optimizer
        device (str): Device to run the training on ("cpu" or "cuda")
        r (int): Rank of LoRA decomposition
        alpha (int): LoRA scaling factor
        dropout (float): Dropout probability in LoRA layers
        task_type (TaskType): PEFT task type
    
    Returns:
        nn.Module: LoRA-adapted fine-tuned model
    """
    
    model.to(device)
    model.train()

    try:
        from slimformers.discovery import DISCOVERY_REGISTRY, default_discover

        cls = type(model).__name__
        finder = DISCOVERY_REGISTRY.get(cls, default_discover)
        blocks = finder(model)

        target_modules = set()
        for blk in blocks:
            if blk["type"] == "ffn":
                target_modules.add(blk["fc_name"].rsplit(".", 1)[-1])
                target_modules.add(blk["proj_name"].rsplit(".", 1)[-1])
            elif blk["type"] == "gated":
                target_modules.add(blk["gate_name"].rsplit(".", 1)[-1])
                target_modules.add(blk["up_name"].rsplit(".", 1)[-1])
                target_modules.add(blk["down_name"].rsplit(".", 1)[-1])
        target_modules = list(sorted(target_modules))

        print(f"[LoRA] Auto-selected target_modules: {target_modules}")
    except Exception as e:
        raise ValueError(f"Could not infer target_modules: {e}")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.get("input_ids")

            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[LoRA] Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

    return model