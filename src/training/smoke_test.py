"""
LEVEL 4.1 - training pipeline smoke test (NOT real training).

Runs the full training path on a small, fixed subset of the dataset:
    Dataset -> DataLoader -> pretrained DeFaX (frozen backbones)
    -> forward -> CrossEntropyLoss -> backward -> AdamW step
    -> validation -> checkpoint save -> checkpoint reload

Only 500 training and 200 validation images, 2 epochs. The accuracy it
prints is meaningless for judging the model; it only proves the system works.

Run from the project root:
    python -m src.training.smoke_test
"""

import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TORCH_HOME = Path(r"D:\FINAL YEAR PROJECT\torch_cache")

sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.hub
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.training.trainer import (
    CLASS_NAMES,
    count_parameters,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_train_mode,
    train_one_epoch,
)

SEED = 42
TRAIN_PER_CLASS = 250        # 500 training images
VALID_PER_CLASS = 100        # 200 validation images
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-3         # head-only training from random init
WEIGHT_DECAY = 1e-2
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "defax_smoke_test.pt"

checks = {}


def check(name, passed, note=""):
    checks[name] = bool(passed)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({note})" if note else ""))
    return bool(passed)


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_subset(dataset, per_class, seed):
    """Deterministically pick `per_class` images of each class (no copying)."""
    targets = torch.tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for label in sorted(CLASS_NAMES):
        class_indices = (targets == label).nonzero(as_tuple=True)[0]
        order = torch.randperm(len(class_indices), generator=generator)
        chosen.extend(class_indices[order[:per_class]].tolist())
    return Subset(dataset, sorted(chosen))


def class_counts(subset):
    labels = [subset.dataset.targets[i] for i in subset.indices]
    return {CLASS_NAMES[c]: labels.count(c) for c in CLASS_NAMES}


def labels_match_folders(subset):
    """Every image under fake/ must be 0 and every image under real/ must be 1."""
    for i in subset.indices:
        path, label = subset.dataset.samples[i]
        folder = Path(path).parent.name
        if (folder == "fake" and label != 0) or (folder == "real" and label != 1):
            return False
    return True


def block_downloads():
    def no_download(url, *args, **kwargs):
        raise RuntimeError(f"Download attempted (blocked by smoke test): {url}")
    torch.hub.download_url_to_file = no_download


def fingerprint(module):
    """Cheap per-tensor summary used to prove frozen weights never change."""
    return [p.detach().double().sum().item() for p in module.parameters()]


def main():
    os.chdir(PROJECT_ROOT)  # dataset.py uses a path relative to the project root
    set_seeds(SEED)
    start = time.perf_counter()

    print("=" * 60)
    print("LEVEL 4.1 TRAINING PIPELINE SMOKE TEST - running checks")
    print("=" * 60)

    # Safety: pretrained weights must come from the D: cache, never downloaded
    hub_dir = Path(torch.hub.get_dir())
    if EXPECTED_TORCH_HOME not in hub_dir.parents:
        print(f"ABORT: torch cache is {hub_dir}, expected under {EXPECTED_TORCH_HOME}.")
        print("Open a NEW terminal so TORCH_HOME is picked up, then rerun.")
        return False
    block_downloads()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = use_cuda
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # ---------------- Data ----------------
    print("\nData")
    try:
        from src.data.dataset import create_datasets  # reuses verified transforms
        train_full, valid_full, _ = create_datasets()
        mapping_ok = train_full.class_to_idx == {"fake": 0, "real": 1} == valid_full.class_to_idx
        train_set = balanced_subset(train_full, TRAIN_PER_CLASS, SEED)
        valid_set = balanced_subset(valid_full, VALID_PER_CLASS, SEED)
        check("Dataset loading", mapping_ok and len(train_set) == 500 and len(valid_set) == 200,
              f"full sets {len(train_full):,}/{len(valid_full):,}, class_to_idx {train_full.class_to_idx}")
    except Exception as error:
        check("Dataset loading", False, f"{type(error).__name__}: {error}")
        return False

    # Same subset on every run: compare against a second, independent selection
    repeat = balanced_subset(train_full, TRAIN_PER_CLASS, SEED)
    deterministic = repeat.indices == train_set.indices
    train_counts, valid_counts = class_counts(train_set), class_counts(valid_set)

    loader_generator = torch.Generator().manual_seed(SEED)
    try:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=use_cuda, generator=loader_generator)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=use_cuda)
        images, labels = next(iter(train_loader))
        check("DataLoader", deterministic,
              f"{len(train_loader)} train / {len(valid_loader)} valid batches; subset selection repeatable")
    except Exception as error:
        check("DataLoader", False, f"{type(error).__name__}: {error}")
        return False

    check("Input shape", tuple(images.shape) == (BATCH_SIZE, 3, 224, 224) and images.dtype == torch.float32,
          f"{list(images.shape)}, {images.dtype}")
    check("Label dtype", labels.dtype == torch.int64, str(labels.dtype))
    check("Label mapping",
          set(labels.unique().tolist()) <= {0, 1}
          and labels_match_folders(train_set) and labels_match_folders(valid_set),
          "only 0/1; every fake/ image is 0 and every real/ image is 1")

    # ---------------- Model ----------------
    print("\nModel")
    try:
        from src.models.defax import DeFaX
        model = DeFaX(pretrained=True, freeze_backbones=True)
        check("Pretrained DeFaX initialization", True, "weights loaded from D: cache, downloads blocked")
    except Exception as error:
        check("Pretrained DeFaX initialization", False, f"{type(error).__name__}: {error}")
        return False

    model = model.to(device)
    check("CUDA model placement",
          all(p.device.type == device.type for p in model.parameters()) and use_cuda,
          str(device) if use_cuda else "CUDA not available")

    trainable, frozen = count_parameters(model)
    check("Swin frozen", not any(p.requires_grad for p in model.swin.parameters()))
    check("EfficientNet frozen", not any(p.requires_grad for p in model.efficientnet.parameters()))
    check("Cross-attention trainable", all(p.requires_grad for p in model.fusion.parameters()))
    check("Classifier trainable", all(p.requires_grad for p in model.classifier.parameters()))

    set_train_mode(model)
    check("Frozen backbones kept in eval mode (extra)",
          not model.swin.training and not model.efficientnet.training
          and model.fusion.training and model.classifier.training,
          "no stochastic depth in frozen Swin")

    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(images.to(device))
    check("Forward pass", tuple(logits.shape) == (BATCH_SIZE, 2), f"logits {list(logits.shape)}")

    # ---------------- Training ----------------
    print(f"\nTraining ({EPOCHS} epochs, {len(train_set)} images)")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    trainable_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    swin_before, eff_before = fingerprint(model.swin), fingerprint(model.efficientnet)

    history = []
    try:
        for epoch in range(1, EPOCHS + 1):
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            valid_stats = evaluate(model, valid_loader, criterion, device, use_amp)
            history.append((train_stats, valid_stats))
            print(f"    epoch {epoch}: train loss {train_stats['loss']:.4f} "
                  f"({train_stats['steps_applied']} steps, {train_stats['steps_skipped']} skipped) | "
                  f"valid loss {valid_stats['loss']:.4f}, accuracy {valid_stats['accuracy']:.1%}")
        training_error = None
    except torch.OutOfMemoryError as error:
        training_error = f"CUDA out of memory: {error}"
    except Exception as error:
        training_error = f"{type(error).__name__}: {error}"

    if training_error:
        print(f"  Training failed: {training_error}")
        for name in ("Loss calculation", "Backward pass", "Gradient check", "Optimizer step",
                     "Parameter update", "Validation", "Validation loss", "Accuracy calculation"):
            check(name, False, "training did not complete")
        return report(train_counts, valid_counts, trainable, frozen, None, None, use_cuda, False)

    all_train = [t for t, _ in history]
    total_applied = sum(t["steps_applied"] for t in all_train)
    total_skipped = sum(t["steps_skipped"] for t in all_train)
    total_steps = total_applied + total_skipped

    check("Loss calculation", all(torch.isfinite(torch.tensor(t["loss"])) for t in all_train),
          "training loss finite every epoch")
    check("Backward pass", total_steps == EPOCHS * len(train_loader),
          f"{total_steps} backward passes")
    check("Gradient check", all(t["applied_with_nonfinite_grads"] == 0 for t in all_train),
          f"all gradients finite on applied steps; {total_skipped} fp16-overflow steps skipped by GradScaler")
    check("Optimizer step", total_applied > 0, f"{total_applied}/{total_steps} steps applied")

    trainable_after = [p.detach() for p in model.parameters() if p.requires_grad]
    changed = sum(not torch.equal(a, b) for a, b in zip(trainable_before, trainable_after))
    frozen_unchanged = (fingerprint(model.swin) == swin_before
                        and fingerprint(model.efficientnet) == eff_before)
    check("Parameter update", changed > 0 and frozen_unchanged,
          f"{changed}/{len(trainable_after)} trainable tensors changed; frozen backbones unchanged")

    final_valid = history[-1][1]
    check("Validation", final_valid["total"] == len(valid_set), f"{final_valid['total']} images evaluated")
    check("Validation loss", torch.isfinite(torch.tensor(final_valid["loss"])).item(),
          f"{final_valid['loss']:.4f}")
    per_class_total = sum(v["total"] for v in final_valid["per_class"].values())
    check("Accuracy calculation",
          0.0 <= final_valid["accuracy"] <= 1.0 and per_class_total == final_valid["total"],
          f"{final_valid['correct']}/{final_valid['total']}")

    # ---------------- Checkpoint ----------------
    print("\nCheckpoint")
    config = {
        "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "freeze_backbones": True, "amp": use_amp,
        "train_samples": len(train_set), "valid_samples": len(valid_set),
        "purpose": "Level 4.1 smoke test - not a usable model",
    }
    try:
        path = save_checkpoint(CHECKPOINT_PATH, model, optimizer, scaler, EPOCHS,
                               {"valid_loss": final_valid["loss"], "valid_accuracy": final_valid["accuracy"]},
                               config)
        size_mb = path.stat().st_size / 1024**2
        check("Checkpoint save", path.is_file() and size_mb > 0, f"{path.relative_to(PROJECT_ROOT)} ({size_mb:.0f} MB)")
    except Exception as error:
        check("Checkpoint save", False, f"{type(error).__name__}: {error}")
        return report(train_counts, valid_counts, trainable, frozen, history, None, use_cuda, False)

    try:
        # Fresh model with RANDOM weights: everything must come from the checkpoint
        reloaded = DeFaX(pretrained=False, freeze_backbones=True).to(device)
        new_optimizer = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad],
                                          lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        new_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        saved = load_checkpoint(CHECKPOINT_PATH, reloaded, new_optimizer, new_scaler, device=device)
        check("Checkpoint reload", saved["epoch"] == EPOCHS and saved["class_names"] == CLASS_NAMES,
              "into a fresh random-weight DeFaX, plus optimizer and scaler state")
    except Exception as error:
        check("Checkpoint reload", False, f"{type(error).__name__}: {error}")
        return report(train_counts, valid_counts, trainable, frozen, history, None, use_cuda, False)

    model.eval()
    reloaded.eval()
    valid_images, _ = next(iter(valid_loader))
    with torch.no_grad():
        original_out = model(valid_images.to(device))
        reloaded_out = reloaded(valid_images.to(device))
    reloaded_valid = evaluate(reloaded, valid_loader, criterion, device, use_amp)
    check("Reloaded forward pass",
          tuple(reloaded_out.shape) == (BATCH_SIZE, 2)
          and torch.isfinite(reloaded_out).all().item()
          and torch.allclose(original_out, reloaded_out, atol=1e-5)
          and reloaded_valid["correct"] == final_valid["correct"],
          "identical outputs and identical validation accuracy to the trained model")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None
    print(f"\nTotal time: {time.perf_counter() - start:.0f}s")
    return report(train_counts, valid_counts, trainable, frozen, history, peak_mb, use_cuda, True)


def report(train_counts, valid_counts, trainable, frozen, history, peak_mb, use_cuda, finished):
    print("\n" + "=" * 60)
    print("LEVEL 4.1 TRAINING PIPELINE SMOKE TEST")
    print("=" * 60)
    print(f"\nDevice: {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {use_cuda}")
    print(f"\nTraining samples: {sum(train_counts.values())}")
    print(f"Validation samples: {sum(valid_counts.values())}")
    print(f"\nFake training samples: {train_counts['FAKE']}")
    print(f"Real training samples: {train_counts['REAL']}")
    print(f"\nFake validation samples: {valid_counts['FAKE']}")
    print(f"Real validation samples: {valid_counts['REAL']}")
    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"\nFrozen parameters: {frozen:,}")
    print(f"Trainable parameters: {trainable:,}\n")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if history:
        train_stats, valid_stats = history[-1]
        print(f"\nTraining loss: {train_stats['loss']:.4f}")
        print(f"Validation loss: {valid_stats['loss']:.4f}")
        print(f"Validation accuracy: {valid_stats['accuracy']:.1%} "
              f"(FAKE {valid_stats['per_class'][0]['correct']}/{valid_stats['per_class'][0]['total']}, "
              f"REAL {valid_stats['per_class'][1]['correct']}/{valid_stats['per_class'][1]['total']})")
        print("  (200-image smoke test after 2 short epochs - NOT a measure of the model)")
    print(f"\nPeak GPU memory: {peak_mb:.0f} MB" if peak_mb is not None else "\nPeak GPU memory: not measured")

    overall = finished and bool(checks) and all(checks.values())
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
