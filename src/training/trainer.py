"""
Reusable training utilities for the DeFaX model.

Mixed precision uses the current API (torch.amp.autocast("cuda") and
torch.amp.GradScaler("cuda")); the old torch.cuda.amp.* calls are deprecated.

Labels: 0 = FAKE, 1 = REAL.
"""

from pathlib import Path

import torch

CLASS_NAMES = {0: "FAKE", 1: "REAL"}


def set_train_mode(model):
    """
    model.train(), but keep fully frozen sub-modules in eval mode.

    A frozen backbone should act as a fixed feature extractor. In train mode
    torchvision's Swin-T applies stochastic depth (randomly skipping layers),
    so its features would change from batch to batch even though its weights
    never update. EfficientNetBranch already handles this itself; this
    function applies the same rule to every frozen part of the model.
    """
    model.train()
    for module in model.children():
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()


def count_parameters(model):
    """Return (trainable, frozen) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def train_step(model, images, labels, criterion, optimizer, scaler, device, use_amp):
    """
    One optimisation step. Returns a dict with:
        loss          float
        grads_finite  all trainable gradients finite after unscaling
        step_applied  False if the GradScaler skipped the step (fp16 overflow)
    """
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(images)
    # Loss in float32 for numerical stability
    loss = criterion(logits.float(), labels)

    scaler.scale(loss).backward()

    # Unscale first so gradients can be inspected at their true size
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    grads_finite = bool(grads) and all(torch.isfinite(g).all() for g in grads)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)   # skipped automatically if gradients are inf/NaN
    scaler.update()
    step_applied = scaler.get_scale() >= scale_before

    return {
        "loss": loss.item(),
        "grads_finite": grads_finite,
        "step_applied": step_applied,
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one pass over `loader`. Returns averaged statistics."""
    set_train_mode(model)
    total_loss = 0.0
    total_samples = 0
    applied = skipped = nonfinite = 0

    for images, labels in loader:
        result = train_step(model, images, labels, criterion, optimizer, scaler, device, use_amp)
        batch = labels.size(0)
        total_loss += result["loss"] * batch
        total_samples += batch
        if result["step_applied"]:
            applied += 1
            if not result["grads_finite"]:
                nonfinite += 1
        else:
            skipped += 1

    return {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "steps_applied": applied,
        "steps_skipped": skipped,           # fp16 overflow, normal early on
        "applied_with_nonfinite_grads": nonfinite,  # must stay 0
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """Validation/test pass. Returns loss, accuracy and per-class counts."""
    model.eval()
    total_loss = 0.0
    total = correct = 0
    per_class = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss = criterion(logits.float(), labels)

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        for c in CLASS_NAMES:
            mask = labels == c
            per_class[c]["total"] += mask.sum().item()
            per_class[c]["correct"] += (predictions[mask] == c).sum().item()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, metrics, config):
    """Save everything needed to resume or evaluate. Creates parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "class_names": CLASS_NAMES,
        },
        path,
    )
    return path


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    """Load a checkpoint into `model` (and optionally optimizer/scaler)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
