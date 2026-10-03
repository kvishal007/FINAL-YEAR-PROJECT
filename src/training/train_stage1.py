"""
LEVEL 4.2 - Stage 1 training runner.

Stage 1 trains only the cross-attention fusion and classifier
(~0.95M parameters) on top of FROZEN pretrained Swin-T and EfficientNet-B0.

Usage (from the project root, in a terminal that has TORCH_HOME set):

    python -m src.training.train_stage1 --dry-run   # checks only, no real training
    python -m src.training.train_stage1             # full 10-epoch Stage 1 run
    python -m src.training.train_stage1 --resume    # continue from last.pt

Outputs (all ignored by Git):
    models/checkpoints/last.pt        state after the latest finished epoch
    models/checkpoints/best.pt        epoch with the lowest validation loss
    results/stage1_history.csv        one row of metrics per epoch

Labels: 0 = FAKE, 1 = REAL.
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.hub
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from src.training.trainer import (
    CLASS_NAMES,
    compute_class_metrics,
    count_parameters,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_train_mode,
    train_one_epoch,
)

CONFIG = {
    "stage": "stage1",
    "seed": 42,
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "scheduler": "cosine",        # per-epoch cosine decay to min_learning_rate; None = constant
    "min_learning_rate": 1e-5,
    "freeze_backbones": True,
    "amp": True,
    "num_workers": 4,             # background image-decoding processes
    "log_every": 250,             # progress line every N training batches
}

EXPECTED_TORCH_HOME = Path(r"D:\FINAL YEAR PROJECT\torch_cache")
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
LAST_PATH = CHECKPOINT_DIR / "last.pt"
BEST_PATH = CHECKPOINT_DIR / "best.pt"
HISTORY_PATH = PROJECT_ROOT / "results" / "stage1_history.csv"
DRY_RUN_DIR = CHECKPOINT_DIR / "dry_run"

EXPECTED_SIZES = {"train": 100_000, "valid": 20_000}

HISTORY_COLUMNS = [
    "epoch", "learning_rate",
    "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    "fake_precision", "fake_recall", "fake_f1",
    "real_precision", "real_recall", "real_f1",
    "train_time_s", "val_time_s", "epoch_time_s", "gpu_peak_mb",
    "steps_skipped", "is_best",
]


# ------------------------------------------------------------------ building blocks

def check_torch_cache():
    """Pretrained weights must come from the D: cache; block any download."""
    hub_dir = Path(torch.hub.get_dir())
    if EXPECTED_TORCH_HOME not in hub_dir.parents:
        raise RuntimeError(
            f"torch cache is {hub_dir}, expected under {EXPECTED_TORCH_HOME}. "
            "Open a NEW terminal so TORCH_HOME is picked up."
        )

    def no_download(url, *args, **kwargs):
        raise RuntimeError(f"Download attempted (blocked): {url}")

    torch.hub.download_url_to_file = no_download
    return hub_dir


def build_datasets():
    from src.data.dataset import create_datasets  # verified data pipeline
    train_set, valid_set, _ = create_datasets()   # test set is not touched in Stage 1
    return train_set, valid_set


def build_loaders(train_set, valid_set, config, use_cuda):
    generator = torch.Generator()
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": use_cuda,
        "persistent_workers": config["num_workers"] > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    valid_loader = DataLoader(valid_set, shuffle=False, **common)
    return train_loader, valid_loader, generator


def build_model(config, device):
    from src.models.defax import DeFaX
    model = DeFaX(pretrained=True, freeze_backbones=config["freeze_backbones"])
    return model.to(device)


def build_optimizer(model, config):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"])
    else:
        scheduler = None
    return optimizer, scheduler


def save_training_state(path, model, optimizer, scheduler, scaler, epoch, config,
                        history, best_val_loss, best_epoch):
    latest = history[-1] if history else {}
    return save_checkpoint(
        path, model, optimizer, scaler, epoch,
        metrics={k: latest.get(k) for k in ("val_loss", "val_accuracy", "fake_f1", "real_f1")},
        config=config,
        extra={
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
    )


def restore_training_state(path, model, optimizer, scheduler, scaler, device, config):
    checkpoint = load_checkpoint(path, model, optimizer, scaler, device=device)
    for key in ("batch_size", "learning_rate", "weight_decay", "epochs",
                "freeze_backbones", "scheduler"):
        if checkpoint["config"].get(key) != config[key]:
            raise RuntimeError(f"Cannot resume: config '{key}' changed "
                               f"({checkpoint['config'].get(key)} -> {config[key]})")
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def write_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in HISTORY_COLUMNS})


def epoch_record(epoch, lr, train_stats, valid_stats, train_time, val_time, peak_mb):
    m = valid_stats["class_metrics"]
    return {
        "epoch": epoch,
        "learning_rate": lr,
        "train_loss": train_stats["loss"],
        "train_accuracy": train_stats["accuracy"],
        "val_loss": valid_stats["loss"],
        "val_accuracy": valid_stats["accuracy"],
        "fake_precision": m["FAKE"]["precision"],
        "fake_recall": m["FAKE"]["recall"],
        "fake_f1": m["FAKE"]["f1"],
        "real_precision": m["REAL"]["precision"],
        "real_recall": m["REAL"]["recall"],
        "real_f1": m["REAL"]["f1"],
        "train_time_s": round(train_time, 1),
        "val_time_s": round(val_time, 1),
        "epoch_time_s": round(train_time + val_time, 1),
        "gpu_peak_mb": round(peak_mb, 0) if peak_mb is not None else None,
        "steps_skipped": train_stats["steps_skipped"],
        "is_best": False,
    }


def print_epoch(record, total_epochs):
    r = record
    print(f"\n  Epoch {r['epoch']}/{total_epochs}  (lr {r['learning_rate']:.2e}, "
          f"{r['epoch_time_s'] / 60:.1f} min, peak GPU {r['gpu_peak_mb']:.0f} MB)"
          + ("  * new best" if r["is_best"] else ""))
    print(f"    train  loss {r['train_loss']:.4f}  acc {r['train_accuracy']:.2%}")
    print(f"    valid  loss {r['val_loss']:.4f}  acc {r['val_accuracy']:.2%}")
    print(f"    FAKE   precision {r['fake_precision']:.4f}  recall {r['fake_recall']:.4f}  F1 {r['fake_f1']:.4f}")
    print(f"    REAL   precision {r['real_precision']:.4f}  recall {r['real_recall']:.4f}  F1 {r['real_f1']:.4f}",
          flush=True)


def git_ignored(path):
    relative = Path(path).resolve().relative_to(PROJECT_ROOT)
    result = subprocess.run(["git", "check-ignore", "-q", str(relative)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def inside(path, folder):
    return Path(path).resolve().is_relative_to(Path(folder).resolve())


# ------------------------------------------------------------------ real training

def train(config, resume):
    torch.manual_seed(config["seed"])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = config["amp"] and use_cuda
    check_torch_cache()

    if not resume and (LAST_PATH.exists() or BEST_PATH.exists()):
        print(f"Refusing to start: {LAST_PATH.name} / {BEST_PATH.name} already exist in {CHECKPOINT_DIR}.")
        print("Use --resume to continue, or move the old checkpoints away first.")
        return False

    train_set, valid_set = build_datasets()
    train_loader, valid_loader, generator = build_loaders(train_set, valid_set, config, use_cuda)
    model = build_model(config, device)
    optimizer, scheduler = build_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    history, best_val_loss, best_epoch, start_epoch = [], math.inf, None, 1
    if resume:
        checkpoint = restore_training_state(LAST_PATH, model, optimizer, scheduler, scaler, device, config)
        history = checkpoint["history"]
        best_val_loss, best_epoch = checkpoint["best_val_loss"], checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {LAST_PATH} after epoch {checkpoint['epoch']} (best so far: epoch {best_epoch})")

    trainable, frozen = count_parameters(model)
    print("=" * 64)
    print("LEVEL 4.2 - STAGE 1 TRAINING")
    print("=" * 64)
    print(f"Device {torch.cuda.get_device_name(0) if use_cuda else 'CPU'} | AMP {use_amp} | "
          f"batch {config['batch_size']} | epochs {start_epoch}-{config['epochs']}")
    print(f"Train {len(train_set):,} | valid {len(valid_set):,} | "
          f"trainable {trainable:,} | frozen {frozen:,}")

    for epoch in range(start_epoch, config["epochs"] + 1):
        generator.manual_seed(config["seed"] + epoch)  # same shuffle order per epoch, even after resume
        lr = optimizer.param_groups[0]["lr"]
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        print(f"\nEpoch {epoch}/{config['epochs']} - training", flush=True)
        t0 = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                      use_amp, log_every=config["log_every"])
        t1 = time.perf_counter()
        print(f"Epoch {epoch}/{config['epochs']} - validating", flush=True)
        valid_stats = evaluate(model, valid_loader, criterion, device, use_amp)
        t2 = time.perf_counter()

        if train_stats["applied_with_nonfinite_grads"] or not math.isfinite(train_stats["loss"]) \
                or not math.isfinite(valid_stats["loss"]):
            print("STOPPING: non-finite loss or gradients detected. last.pt was NOT overwritten.")
            return False

        peak = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None
        record = epoch_record(epoch, lr, train_stats, valid_stats, t1 - t0, t2 - t1, peak)
        if valid_stats["loss"] < best_val_loss:
            best_val_loss, best_epoch = valid_stats["loss"], epoch
            record["is_best"] = True
        history.append(record)
        if scheduler is not None:
            scheduler.step()

        save_training_state(LAST_PATH, model, optimizer, scheduler, scaler, epoch, config,
                            history, best_val_loss, best_epoch)
        if record["is_best"]:
            save_training_state(BEST_PATH, model, optimizer, scheduler, scaler, epoch, config,
                                history, best_val_loss, best_epoch)
        write_history(history, HISTORY_PATH)
        print_epoch(record, config["epochs"])

    print("\n" + "=" * 64)
    print(f"STAGE 1 COMPLETE - best epoch {best_epoch}, validation loss {best_val_loss:.4f}")
    print(f"  {BEST_PATH}\n  {LAST_PATH}\n  {HISTORY_PATH}")
    print("Validation metrics guide model selection; the test set has not been used.")
    print("=" * 64)
    return True


# ------------------------------------------------------------------ dry run

DRY_WARMUP_BATCHES = 5
DRY_TRAIN_BATCHES = 40
DRY_VAL_BATCHES = 20


def dry_run(config):
    """Check every part of the real run without training on the full dataset."""
    warnings.simplefilter("error", FutureWarning)  # fail on deprecated APIs (e.g. old AMP)
    torch.manual_seed(config["seed"])
    results = {}

    def check(name, passed, note=""):
        results[name] = bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({note})" if note else ""), flush=True)
        return bool(passed)

    print("=" * 64)
    print("LEVEL 4.2 - STAGE 1 DRY RUN (no real training)")
    print("=" * 64)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = config["amp"] and use_cuda

    # 1. Configuration
    print("\nConfiguration")
    config_ok = (config["epochs"] == 10 and config["batch_size"] == 32 and config["amp"]
                 and config["freeze_backbones"] and CLASS_NAMES == {0: "FAKE", 1: "REAL"})
    check("Configuration", config_ok,
          f"epochs {config['epochs']}, batch {config['batch_size']}, lr {config['learning_rate']}, "
          f"wd {config['weight_decay']}, scheduler {config['scheduler']}, workers {config['num_workers']}")
    check("CUDA + AMP", use_cuda and use_amp, torch.cuda.get_device_name(0) if use_cuda else "no CUDA")
    try:
        hub_dir = check_torch_cache()
        check("Pretrained weight cache", True, f"{hub_dir}, downloads blocked")
    except RuntimeError as error:
        check("Pretrained weight cache", False, str(error))
        return report_dry_run(results, None)

    # 2. Data
    print("\nData")
    train_set, valid_set = build_datasets()
    sizes_ok = len(train_set) == EXPECTED_SIZES["train"] and len(valid_set) == EXPECTED_SIZES["valid"]
    balance = {split: {CLASS_NAMES[c]: ds.targets.count(c) for c in CLASS_NAMES}
               for split, ds in (("train", train_set), ("valid", valid_set))}
    check("Full datasets", sizes_ok and train_set.class_to_idx == {"fake": 0, "real": 1} == valid_set.class_to_idx,
          f"train {len(train_set):,} {balance['train']}, valid {len(valid_set):,} {balance['valid']}")

    train_loader, valid_loader, generator = build_loaders(train_set, valid_set, config, use_cuda)
    generator.manual_seed(config["seed"] + 1)
    images, labels = next(iter(train_loader))
    check("DataLoaders",
          len(train_loader) == 3125 and len(valid_loader) == 625
          and isinstance(train_loader.sampler, RandomSampler)
          and isinstance(valid_loader.sampler, SequentialSampler)
          and tuple(images.shape) == (32, 3, 224, 224) and images.dtype == torch.float32
          and labels.dtype == torch.int64 and set(labels.unique().tolist()) <= {0, 1},
          f"{len(train_loader)} train batches (shuffled) / {len(valid_loader)} valid batches "
          f"(fixed order), {config['num_workers']} workers, batch {list(images.shape)}")

    # 3. Model
    print("\nModel")
    model = build_model(config, device)
    trainable, frozen = count_parameters(model)
    set_train_mode(model)
    check("Pretrained DeFaX", trainable == 952_194 and frozen == 31_526_902
          and all(p.device.type == device.type for p in model.parameters()),
          f"trainable {trainable:,}, frozen {frozen:,}, on {device}")
    check("Backbones frozen and in eval mode",
          not any(p.requires_grad for p in model.swin.parameters())
          and not any(p.requires_grad for p in model.efficientnet.parameters())
          and not model.swin.training and not model.efficientnet.training
          and model.fusion.training and model.classifier.training)

    # 4. Optimizer, scheduler, AMP
    print("\nOptimizer / scheduler / AMP")
    optimizer, scheduler = build_optimizer(model, config)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    check("Optimizer", optimized == expected and type(optimizer).__name__ == "AdamW",
          f"AdamW over exactly the {len(expected)} trainable tensors")

    lrs = []
    preview_opt, preview_sched = build_optimizer(build_model_stub(), config)
    for _ in range(config["epochs"]):
        lrs.append(preview_opt.param_groups[0]["lr"])
        preview_opt.step()
        if preview_sched:
            preview_sched.step()
    check("Learning-rate schedule", lrs[0] == config["learning_rate"] and lrs == sorted(lrs, reverse=True),
          "epoch lrs " + ", ".join(f"{lr:.1e}" for lr in lrs))

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        amp_logits = model(images[:4].to(device))
    check("AMP", scaler.is_enabled() and amp_logits.dtype == torch.float16,
          "torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda'); no deprecated API warnings")

    # 5. Metrics function on a known confusion matrix
    known = compute_class_metrics([[8, 2], [1, 9]])  # rows = true FAKE/REAL, cols = predicted
    metric_ok = (abs(known["FAKE"]["precision"] - 8 / 9) < 1e-9 and abs(known["FAKE"]["recall"] - 0.8) < 1e-9
                 and abs(known["REAL"]["precision"] - 9 / 11) < 1e-9 and abs(known["REAL"]["recall"] - 0.9) < 1e-9)
    check("Precision/recall/F1 formula", metric_ok, "verified against a hand-calculated example")

    # 6. Short training + validation on the REAL loaders (timed)
    print(f"\nShort run ({DRY_TRAIN_BATCHES} train + {DRY_VAL_BATCHES} valid batches)")
    criterion = nn.CrossEntropyLoss()
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    frozen_before = [p.detach().double().sum().item() for p in model.swin.parameters()]
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                    max_batches=DRY_WARMUP_BATCHES)  # warm-up: worker start, kernel setup
    t0 = time.perf_counter()
    train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                                  max_batches=DRY_TRAIN_BATCHES)
    t1 = time.perf_counter()

    # The real validation loader is in folder order (all 10,000 fake/ images
    # first), so its first batches contain only FAKE. Use a small balanced
    # slice here so both classes' metrics are actually exercised.
    per_class = DRY_VAL_BATCHES * config["batch_size"] // 2
    balanced = [i for c in CLASS_NAMES
                for i in [j for j, t in enumerate(valid_set.targets) if t == c][:per_class]]
    dry_valid_loader = DataLoader(Subset(valid_set, balanced), batch_size=config["batch_size"],
                                  shuffle=False, num_workers=config["num_workers"],
                                  pin_memory=use_cuda, persistent_workers=config["num_workers"] > 0)
    evaluate(model, dry_valid_loader, criterion, device, use_amp, max_batches=DRY_WARMUP_BATCHES)
    t2 = time.perf_counter()
    valid_stats = evaluate(model, dry_valid_loader, criterion, device, use_amp)
    t3 = time.perf_counter()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None

    changed = sum(not torch.equal(a, b) for a, b in
                  zip(before, [p.detach() for p in model.parameters() if p.requires_grad]))
    frozen_same = frozen_before == [p.detach().double().sum().item() for p in model.swin.parameters()]
    check("Training steps",
          math.isfinite(train_stats["loss"]) and train_stats["applied_with_nonfinite_grads"] == 0
          and train_stats["steps_applied"] > 0 and changed > 0 and frozen_same,
          f"loss {train_stats['loss']:.4f}, acc {train_stats['accuracy']:.1%}, "
          f"{train_stats['steps_applied']} steps applied, {changed} tensors updated, frozen Swin unchanged")

    m = valid_stats["class_metrics"]
    values = [v for cls in m.values() for v in cls.values()]
    seen = [sum(row) for row in valid_stats["confusion"]]  # images per true class
    check("Validation metrics",
          math.isfinite(valid_stats["loss"]) and all(0.0 <= v <= 1.0 for v in values)
          and sum(seen) == valid_stats["total"] and seen == [per_class, per_class],
          f"{valid_stats['total']} images ({seen[0]} FAKE / {seen[1]} REAL), "
          f"loss {valid_stats['loss']:.4f}, acc {valid_stats['accuracy']:.1%}, "
          f"FAKE P/R/F1 {m['FAKE']['precision']:.3f}/{m['FAKE']['recall']:.3f}/{m['FAKE']['f1']:.3f}, "
          f"REAL P/R/F1 {m['REAL']['precision']:.3f}/{m['REAL']['recall']:.3f}/{m['REAL']['f1']:.3f}")

    # 7. Checkpoint paths and Git
    print("\nCheckpoints")
    paths = {"last.pt": LAST_PATH, "best.pt": BEST_PATH, "history CSV": HISTORY_PATH}
    check("Paths outside the dataset", not any(inside(p, DATA_DIR) for p in paths.values())
          and all(inside(p, PROJECT_ROOT) for p in paths.values()),
          ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in paths.values()))
    temporary = LAST_PATH.with_name(LAST_PATH.name + ".tmp")
    ignored = {name: git_ignored(p) for name, p in {**paths, "temporary .tmp": temporary}.items()}
    check(".gitignore excludes outputs", all(ignored.values()),
          ", ".join(f"{k}: {'ignored' if v else 'NOT ignored'}" for k, v in ignored.items()))
    free_gb = shutil.disk_usage(CHECKPOINT_DIR.parent).free / 1024**3
    check("Disk space", free_gb > 5, f"{free_gb:.0f} GB free on {CHECKPOINT_DIR.drive}")
    existing = [p.name for p in (LAST_PATH, BEST_PATH) if p.exists()]
    check("No old Stage 1 checkpoints to overwrite", not existing,
          "none present" if not existing else f"found {existing} - a new run would refuse to start")

    # 8. Save / reload / resume round trip (separate folder, removed afterwards)
    history = [epoch_record(1, lrs[0], train_stats, valid_stats, t1 - t0, t3 - t2, peak_mb)]
    history[0]["is_best"] = True
    if scheduler:
        scheduler.step()
    try:
        dry_last = save_training_state(DRY_RUN_DIR / "last.pt", model, optimizer, scheduler, scaler, 1,
                                       config, history, history[0]["val_loss"], 1)
        dry_best = save_training_state(DRY_RUN_DIR / "best.pt", model, optimizer, scheduler, scaler, 1,
                                       config, history, history[0]["val_loss"], 1)
        write_history(history, DRY_RUN_DIR / "history.csv")

        fresh = build_model_stub().to(device)  # random weights - everything must come from the file
        fresh_opt, fresh_sched = build_optimizer(fresh, config)
        fresh_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        restored = restore_training_state(dry_last, fresh, fresh_opt, fresh_sched, fresh_scaler, device, config)

        model.eval(); fresh.eval()
        with torch.no_grad():
            same_output = torch.allclose(model(images[:8].to(device)), fresh(images[:8].to(device)), atol=1e-5)
        same_lr = fresh_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        same_step = (fresh_opt.state_dict()["state"][0]["step"] == optimizer.state_dict()["state"][0]["step"])
        no_tmp = not any(DRY_RUN_DIR.glob("*.tmp"))
        size_mb = dry_last.stat().st_size / 1024**2
        check("Checkpoint save / reload / resume",
              same_output and same_lr and same_step and no_tmp and restored["epoch"] == 1
              and restored["history"][0]["epoch"] == 1 and dry_best.is_file(),
              f"{size_mb:.0f} MB each; model, optimizer, scheduler, scaler and history restored; "
              f"resume would start at epoch {restored['epoch'] + 1}")
    except Exception as error:
        check("Checkpoint save / reload / resume", False, f"{type(error).__name__}: {error}")
    finally:
        shutil.rmtree(DRY_RUN_DIR, ignore_errors=True)

    # 9. Time estimate from the measured throughput
    train_per_batch = (t1 - t0) / DRY_TRAIN_BATCHES
    val_per_batch = (t3 - t2) / DRY_VAL_BATCHES
    epoch_min = (train_per_batch * len(train_loader) + val_per_batch * len(valid_loader)) / 60
    estimate = {
        "train_s_per_batch": train_per_batch,
        "val_s_per_batch": val_per_batch,
        "epoch_min": epoch_min,
        "total_hours": epoch_min * config["epochs"] / 60,
        "peak_mb": peak_mb,
    }
    return report_dry_run(results, estimate)


def build_model_stub():
    """DeFaX with random weights (no cache needed) - used for schedule preview and reload test."""
    from src.models.defax import DeFaX
    return DeFaX(pretrained=False, freeze_backbones=CONFIG["freeze_backbones"])


def report_dry_run(results, estimate):
    print("\n" + "=" * 64)
    print("DRY RUN SUMMARY")
    print("=" * 64)
    for name, passed in results.items():
        print(f"{name:<42}: {'PASS' if passed else 'FAIL'}")
    if estimate:
        print(f"\nMeasured speed      : {estimate['train_s_per_batch']:.3f} s/train batch, "
              f"{estimate['val_s_per_batch']:.3f} s/valid batch")
        print(f"Estimated epoch     : ~{estimate['epoch_min']:.0f} min (3125 train + 625 valid batches)")
        print(f"Estimated 10 epochs : ~{estimate['total_hours']:.1f} h")
        if estimate["peak_mb"] is not None:
            print(f"Peak GPU memory     : {estimate['peak_mb']:.0f} MB")
    ready = bool(results) and all(results.values())
    print("\n" + "=" * 64)
    print("READY" if ready else "NOT READY")
    print("=" * 64)
    return ready


def main():
    parser = argparse.ArgumentParser(description="DeFaX Stage 1 training (frozen backbones)")
    parser.add_argument("--dry-run", action="store_true", help="run all checks without real training")
    parser.add_argument("--resume", action="store_true", help="continue from models/checkpoints/last.pt")
    args = parser.parse_args()

    if args.dry_run:
        return dry_run(CONFIG)
    return train(CONFIG, resume=args.resume)


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    sys.exit(0 if main() else 1)
