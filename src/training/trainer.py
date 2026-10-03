"""
Reusable training utilities for the DeFaX model.

Mixed precision uses the current API (torch.amp.autocast("cuda") and
torch.amp.GradScaler("cuda")); the old torch.cuda.amp.* calls are deprecated.

Labels: 0 = FAKE, 1 = REAL.
"""

import os
import time
from pathlib import Path

import torch

CLASS_NAMES = {0: "FAKE", 1: "REAL"}


def compute_class_metrics(confusion):
    """
    Per-class precision, recall and F1 from a confusion matrix.

    confusion[true_label][predicted_label] = count
    Each class is treated in turn as the "positive" class, so the FAKE
    metrics answer "how well are fakes detected?". Undefined ratios
    (division by zero) are reported as 0.0.
    """
    confusion = torch.as_tensor(confusion, dtype=torch.float64)
    metrics = {}
    for c, name in CLASS_NAMES.items():
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp   # predicted c, actually other
        fn = confusion[c, :].sum().item() - tp   # actually c, predicted other
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def set_train_mode(model, freeze_batchnorm=False):
    """
    model.train(), but keep fully frozen sub-modules in eval mode.

    A frozen backbone should act as a fixed feature extractor. In train mode
    torchvision's Swin-T applies stochastic depth (randomly skipping layers),
    so its features would change from batch to batch even though its weights
    never update. EfficientNetBranch already handles this itself; this
    function applies the same rule to every frozen part of the model.

    freeze_batchnorm: keep every BatchNorm layer in eval mode, so its running
    mean/variance stay fixed while its weight and bias still train. Used for
    small-batch fine-tuning, where per-batch statistics are too noisy.
    """
    model.train()
    for module in model.children():
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def count_parameters(model):
    """Return (trainable, frozen) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def train_step(model, images, labels, criterion, optimizer, scaler, device, use_amp,
               max_grad_norm=None):
    """
    One optimisation step. max_grad_norm (optional) clips the total gradient
    norm after unscaling. Returns a dict with:
        loss          float
        correct       number of correct predictions in the batch
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
    if max_grad_norm is not None and grads_finite:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)   # skipped automatically if gradients are inf/NaN
    scaler.update()
    step_applied = scaler.get_scale() >= scale_before

    return {
        "loss": loss.item(),
        "correct": (logits.argmax(dim=1) == labels).sum().item(),
        "grads_finite": grads_finite,
        "step_applied": step_applied,
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    max_batches=None, log_every=None, max_grad_norm=None,
                    freeze_batchnorm=False):
    """
    Train for one pass over `loader`. Returns averaged statistics.

    max_batches:      stop early after this many batches (dry runs).
    log_every:        print progress every N batches (long epochs).
    max_grad_norm:    optional gradient clipping (see train_step).
    freeze_batchnorm: keep BatchNorm running statistics fixed (see set_train_mode).
    Accuracy is measured on the fly in train mode (dropout active).
    """
    set_train_mode(model, freeze_batchnorm=freeze_batchnorm)
    total_loss = 0.0
    total_samples = total_correct = 0
    applied = skipped = nonfinite = 0
    num_batches = min(len(loader), max_batches) if max_batches else len(loader)
    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader, start=1):
        result = train_step(model, images, labels, criterion, optimizer, scaler, device, use_amp,
                            max_grad_norm=max_grad_norm)
        batch = labels.size(0)
        total_loss += result["loss"] * batch
        total_correct += result["correct"]
        total_samples += batch
        if result["step_applied"]:
            applied += 1
            if not result["grads_finite"]:
                nonfinite += 1
        else:
            skipped += 1

        if log_every and batch_index % log_every == 0:
            elapsed = time.perf_counter() - start
            remaining = elapsed / batch_index * (num_batches - batch_index)
            print(f"    batch {batch_index:>5}/{num_batches}  loss {total_loss / total_samples:.4f}  "
                  f"acc {total_correct / total_samples:.2%}  "
                  f"{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left", flush=True)
        if max_batches and batch_index >= max_batches:
            break

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
        "steps_applied": applied,
        "steps_skipped": skipped,           # fp16 overflow, normal early on
        "applied_with_nonfinite_grads": nonfinite,  # must stay 0
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, max_batches=None):
    """
    Validation/test pass. Returns loss, accuracy, per-class counts, the
    confusion matrix (confusion[true][predicted]) and per-class
    precision/recall/F1 ("class_metrics").
    """
    model.eval()
    total_loss = 0.0
    total = correct = 0
    per_class = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}
    num_classes = len(CLASS_NAMES)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch_index, (images, labels) in enumerate(loader, start=1):
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
        confusion += torch.bincount(
            labels.cpu() * num_classes + predictions.cpu(), minlength=num_classes ** 2
        ).view(num_classes, num_classes)

        if max_batches and batch_index >= max_batches:
            break

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "class_metrics": compute_class_metrics(confusion),
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, metrics, config, extra=None):
    """
    Save everything needed to resume or evaluate. Creates parent folders.

    The file is written to a temporary name first and then renamed, so an
    interruption (crash, sleep, power loss) never leaves a half-written
    checkpoint in place of a good one.
    `extra` adds more entries (e.g. scheduler state, training history).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "class_names": CLASS_NAMES,
    }
    if extra:
        checkpoint.update(extra)

    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)  # atomic on the same drive
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
