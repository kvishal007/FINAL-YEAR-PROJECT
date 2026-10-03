"""
LEVEL 4.6 - Stage 2 full fine-tuning runner.

Starts from the Stage 1 best checkpoint (read only) and fine-tunes the WHOLE
DeFaX model (Swin-T, EfficientNet-B0, cross-attention, classifier) on the
full training split, selecting the best epoch by validation loss.

The test split is LOCKED: it is never listed or opened (TestSplitGuard is
active in this process and in every DataLoader worker). No test metrics are
computed here.

Usage (from the project root, in a terminal that has TORCH_HOME set):

    python -m src.training.train_stage2 --dry-run   # checks only, no real training
    python -m src.training.train_stage2             # full Stage 2 run
    python -m src.training.train_stage2 --resume    # continue from Stage 2 last.pt

Outputs (all ignored by Git, all inside the Stage 2 folders):
    models/checkpoints/stage2/last.pt      state after the latest finished epoch
    models/checkpoints/stage2/best.pt      epoch with the lowest validation loss
    results/stage2/history.csv             one row of metrics per epoch
    results/stage2/run_config.json         configuration + Stage 1 starting point
    results/stage2/train_log.txt           everything printed during training
    results/stage2/final_summary.json      best epoch and its validation metrics

Labels: 0 = FAKE, 1 = REAL.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

# Must be set before CUDA/cuBLAS initialise, for deterministic matrix multiplies
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

from src.data.transforms import train_transform, valid_test_transform
from src.evaluation.evaluate_test import sha256
from src.models.defax import DeFaX
from src.training.smoke_test import balanced_subset, class_counts
from src.training.stage2 import (
    PROTECTED_PATHS,
    STAGE1_BEST,
    STAGE2_CHECKPOINT_DIR,
    STAGE2_CONFIG,
    STAGE2_RESULTS_DIR,
    TEST_DIR,
    TestSplitGuard,
    assert_stage2_output,
    build_datasets,
    build_loader,
    build_model_from_stage1,
    build_optimizer,
    changed,
    guard_worker,
    random_operations,
    save_stage2_checkpoint,
    snapshot,
    stage1_artifact_files,
    weights_fingerprint,
)
from src.training.train_stage1 import epoch_record, git_ignored, print_epoch, write_history
from src.training.trainer import (
    CLASS_NAMES,
    count_parameters,
    evaluate,
    load_checkpoint,
    set_train_mode,
    train_one_epoch,
    train_step,
)

LAST_PATH = STAGE2_CHECKPOINT_DIR / "last.pt"
BEST_PATH = STAGE2_CHECKPOINT_DIR / "best.pt"
HISTORY_PATH = STAGE2_RESULTS_DIR / "history.csv"
CONFIG_PATH = STAGE2_RESULTS_DIR / "run_config.json"
LOG_PATH = STAGE2_RESULTS_DIR / "train_log.txt"
SUMMARY_PATH = STAGE2_RESULTS_DIR / "final_summary.json"
DRY_RUN_DIR = STAGE2_CHECKPOINT_DIR / "dry_run"
DRY_RUN_CHECKPOINT = DRY_RUN_DIR / "dry_run.pt"
DRY_SUMMARY_PATH = STAGE2_RESULTS_DIR / "dry_run_summary.json"
OUTPUT_PATHS = [LAST_PATH, BEST_PATH, HISTORY_PATH, CONFIG_PATH, LOG_PATH, SUMMARY_PATH]

EXPECTED_SIZES = {"train": 100_000, "valid": 20_000}
EXPECTED_TRAINABLE = 32_479_096
# Settings that must not change between the original run and a --resume
RESUME_KEYS = ("stage", "seed", "init_checkpoint", "epochs", "batch_size", "learning_rate",
               "weight_decay", "scheduler", "min_learning_rate", "max_grad_norm",
               "freeze_batchnorm", "freeze_backbones", "amp")


# ------------------------------------------------------------------ shared setup

def setup(config):
    """Seeds, deterministic algorithms, device and the main-process test guard."""
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    guard = TestSplitGuard.install()
    guard.active = True
    try:
        os.listdir(TEST_DIR)
        raise RuntimeError("Test-split guard is not blocking access")
    except PermissionError:
        guard.blocked.clear()  # the self-test block is expected

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return guard, device, use_cuda, config["amp"] and use_cuda


def build_loaders(train_set, valid_set, config, use_cuda):
    generator = torch.Generator()
    train_loader = build_loader(train_set, config["batch_size"], True, config["num_workers"],
                                use_cuda, generator, persistent_workers=False)
    valid_loader = build_loader(valid_set, config["eval_batch_size"], False,
                                config["num_workers"], use_cuda)
    return train_loader, valid_loader, generator


def empty_model(config, device):
    """Random-weight DeFaX for --resume: every weight then comes from Stage 2 last.pt."""
    return DeFaX(pretrained=False, freeze_backbones=config["freeze_backbones"]).to(device)


def restore_stage2(path, model, optimizer, scheduler, scaler, device, config):
    """Load a STAGE 2 checkpoint only; Stage 1 checkpoints are refused."""
    path = Path(path).resolve()
    if not path.is_relative_to(STAGE2_CHECKPOINT_DIR.resolve()):
        raise PermissionError(f"--resume only loads Stage 2 checkpoints from {STAGE2_CHECKPOINT_DIR}, not {path}")
    checkpoint = load_checkpoint(path, model, optimizer, scaler, device=device)
    saved = checkpoint["config"]
    if saved.get("stage") != "stage2":
        raise RuntimeError(f"{path.name} is not a Stage 2 checkpoint (stage={saved.get('stage')})")
    for key in RESUME_KEYS:
        if saved.get(key) != config[key]:
            raise RuntimeError(f"Cannot resume: config '{key}' changed ({saved.get(key)} -> {config[key]})")
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def write_json(path, data):
    path = assert_stage2_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fingerprint_stage1():
    return {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in stage1_artifact_files()}


class Tee:
    """Send everything printed to the console AND to the training log."""

    def __init__(self, path):
        path = assert_stage2_output(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.console = sys.stdout

    def write(self, text):
        self.console.write(text)
        self.file.write(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def __getattr__(self, name):  # encoding, isatty, ... come from the real console
        return getattr(self.console, name)


# ------------------------------------------------------------------ real training

def train(config, resume):
    # Decide fresh vs resume BEFORE touching anything
    if resume and not LAST_PATH.exists():
        print(f"Cannot resume: {LAST_PATH} does not exist. Start a fresh run without --resume.")
        return False
    if not resume and any(p.exists() for p in (LAST_PATH, BEST_PATH, HISTORY_PATH)):
        print(f"Refusing to start: Stage 2 outputs already exist in {STAGE2_CHECKPOINT_DIR} / "
              f"{STAGE2_RESULTS_DIR}. Use --resume, or move them away first.")
        return False

    sys.stdout = Tee(LOG_PATH)
    guard, device, use_cuda, use_amp = setup(config)
    stage1_before = fingerprint_stage1()
    run_start = time.perf_counter()

    print("\n" + "=" * 64)
    print(f"LEVEL 4.6 - STAGE 2 TRAINING ({'resume' if resume else 'fresh run'}) "
          f"{datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 64)

    train_set, valid_set = build_datasets()
    if len(train_set) != EXPECTED_SIZES["train"] or len(valid_set) != EXPECTED_SIZES["valid"]:
        raise RuntimeError(f"Unexpected dataset sizes {len(train_set)} / {len(valid_set)}")
    train_loader, valid_loader, generator = build_loaders(train_set, valid_set, config, use_cuda)

    if resume:
        model = empty_model(config, device)
        optimizer, scheduler = build_optimizer(model, config)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        checkpoint = restore_stage2(LAST_PATH, model, optimizer, scheduler, scaler, device, config)
        history = checkpoint["history"]
        best_val_loss, best_epoch = checkpoint["best_val_loss"], checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed Stage 2 from {LAST_PATH.relative_to(PROJECT_ROOT)} after epoch "
              f"{checkpoint['epoch']} (best so far: epoch {best_epoch}, val loss {best_val_loss:.4f})")
    else:
        model, stage1_checkpoint = build_model_from_stage1(config, device)
        optimizer, scheduler = build_optimizer(model, config)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        history, best_val_loss, best_epoch, start_epoch = [], math.inf, None, 1
        write_json(CONFIG_PATH, {
            "config": config,
            "init_checkpoint_sha256": sha256(STAGE1_BEST),
            "init_checkpoint_epoch": stage1_checkpoint["epoch"],
            "init_checkpoint_val_loss": stage1_checkpoint["metrics"]["val_loss"],
            "train_images": len(train_set),
            "valid_images": len(valid_set),
            "label_map": {str(k): v for k, v in CLASS_NAMES.items()},
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        print(f"Initialised from Stage 1 {config['init_checkpoint']} (epoch {stage1_checkpoint['epoch']}, "
              f"val loss {stage1_checkpoint['metrics']['val_loss']:.4f})")

    trainable, frozen = count_parameters(model)
    if trainable != EXPECTED_TRAINABLE or frozen != 0:
        raise RuntimeError(f"Expected {EXPECTED_TRAINABLE:,} trainable / 0 frozen, got {trainable:,} / {frozen:,}")
    print(f"Device {torch.cuda.get_device_name(0) if use_cuda else 'CPU'} | AMP {use_amp} | "
          f"batch {config['batch_size']} | lr {config['learning_rate']:.0e} | epochs "
          f"{start_epoch}-{config['epochs']} | workers {config['num_workers']}")
    print(f"Train {len(train_set):,} | valid {len(valid_set):,} | trainable {trainable:,} | frozen {frozen:,}")

    for epoch in range(start_epoch, config["epochs"] + 1):
        generator.manual_seed(config["seed"] + epoch)  # same order/augmentation per epoch, even after resume
        lr = optimizer.param_groups[0]["lr"]
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        print(f"\nEpoch {epoch}/{config['epochs']} - training", flush=True)
        t0 = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(), optimizer, scaler,
                                      device, use_amp, log_every=config["log_every"],
                                      max_grad_norm=config["max_grad_norm"],
                                      freeze_batchnorm=config["freeze_batchnorm"])
        t1 = time.perf_counter()
        print(f"Epoch {epoch}/{config['epochs']} - validating", flush=True)
        valid_stats = evaluate(model, valid_loader, nn.CrossEntropyLoss(), device, use_amp)
        t2 = time.perf_counter()

        if train_stats["applied_with_nonfinite_grads"] or not math.isfinite(train_stats["loss"]) \
                or not math.isfinite(valid_stats["loss"]):
            print("STOPPING: non-finite loss or gradients. Stage 2 last.pt was NOT overwritten.")
            return False

        peak = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None
        record = epoch_record(epoch, lr, train_stats, valid_stats, t1 - t0, t2 - t1, peak)
        if valid_stats["loss"] < best_val_loss:
            best_val_loss, best_epoch = valid_stats["loss"], epoch
            record["is_best"] = True
        history.append(record)
        if scheduler is not None:
            scheduler.step()

        save_stage2_checkpoint(LAST_PATH, model, optimizer, scheduler, scaler, epoch, config,
                               history, best_val_loss, best_epoch)
        if record["is_best"]:
            save_stage2_checkpoint(BEST_PATH, model, optimizer, scheduler, scaler, epoch, config,
                                   history, best_val_loss, best_epoch)
        write_history(history, assert_stage2_output(HISTORY_PATH))
        print_epoch(record, config["epochs"])

    best = next(r for r in history if r["epoch"] == best_epoch)
    stage1_ok = fingerprint_stage1() == stage1_before
    write_json(SUMMARY_PATH, {
        "description": "Stage 2 fine-tuning - validation results only (test set locked)",
        "best_epoch": best_epoch,
        "best_validation": best,
        "epochs_completed": len(history),
        "run_time_hours_this_session": round((time.perf_counter() - run_start) / 3600, 2),
        "stage1_artifacts_unchanged": stage1_ok,
        "test_split_accesses": len(guard.blocked),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    print("\n" + "=" * 64)
    print(f"STAGE 2 COMPLETE - best epoch {best_epoch}, validation loss {best_val_loss:.4f}, "
          f"validation accuracy {best['val_accuracy']:.2%}")
    print(f"Stage 1 artifacts unchanged: {stage1_ok} | test-split accesses: {len(guard.blocked)}")
    for path in OUTPUT_PATHS:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print("The test set has NOT been evaluated. Do that once, after choosing the final checkpoint.")
    print("=" * 64)
    return stage1_ok and not guard.blocked


# ------------------------------------------------------------------ dry run

DRY_WARMUP_STEPS = 3     # first fp16 step per batch shape compiles kernels (slow)
DRY_TIMED_STEPS = 60     # one continuous pass; each step timed individually
DRY_STARTUP_STEPS = 10   # excluded from the speed estimate (worker start-up, prefetch fill)
DRY_VALID_PER_CLASS = 64


def dry_run(config):
    """Every check of the real run, on a few batches. Never creates last.pt / best.pt."""
    warnings.simplefilter("error", FutureWarning)  # fail on deprecated APIs (e.g. old AMP)
    guard, device, use_cuda, use_amp = setup(config)
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2 if use_cuda else None
    results, facts = {}, {}
    stage2_existing = [p.name for p in (LAST_PATH, BEST_PATH) if p.exists()]

    def check(name, passed, note=""):
        results[name] = bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({note})" if note else ""), flush=True)

    print("=" * 64)
    print("LEVEL 4.6 - STAGE 2 DRY RUN (no full training)")
    print("=" * 64)

    # 1. Configuration and safety
    print("\nConfiguration and safety")
    expected = {"batch_size": 8, "learning_rate": 1e-5, "weight_decay": 0.01, "scheduler": "cosine",
                "max_grad_norm": 1.0, "epochs": 5, "amp": True, "freeze_batchnorm": True,
                "freeze_backbones": False, "num_workers": 4, "init_checkpoint": "models/checkpoints/best.pt"}
    mismatched = {k: config[k] for k, v in expected.items() if config[k] != v}
    check("Configuration", not mismatched and use_amp,
          "batch 8, AdamW lr 1e-5 wd 0.01, cosine, clip 1.0, 5 epochs, AMP, BN stats frozen, 4 workers"
          if not mismatched else f"unexpected: {mismatched}")
    check("Test split blocked (main process)", guard.active, "deliberate listing of test/ was blocked")

    stage1_before = fingerprint_stage1()
    refused = 0
    for target in [*PROTECTED_PATHS, PROJECT_ROOT / "models" / "checkpoints" / "x.pt", PROJECT_ROOT / "data" / "x"]:
        try:
            assert_stage2_output(target)
        except PermissionError:
            refused += 1
    try:
        restore_stage2(STAGE1_BEST, None, None, None, None, device, config)
        resume_refuses_stage1 = False
    except PermissionError:
        resume_refuses_stage1 = True
    check("Stage 1 write protection", refused == len(PROTECTED_PATHS) + 2 and resume_refuses_stage1,
          f"{refused} protected/out-of-scope paths refused; --resume refuses Stage 1 best.pt")

    paths_ok = all(assert_stage2_output(p) and git_ignored(p) for p in OUTPUT_PATHS + [DRY_RUN_CHECKPOINT])
    check("Checkpoint/result paths", paths_ok and not stage2_existing,
          "all under models/checkpoints/stage2 or results/stage2 and git-ignored; "
          + ("no Stage 2 last.pt/best.pt yet" if not stage2_existing else f"EXISTING: {stage2_existing}"))

    # 2. Data
    print("\nData")
    train_set, valid_set = build_datasets()  # train/ and valid/ only; split membership verified
    distribution = {name: class_counts_full(ds) for name, ds in (("train", train_set), ("valid", valid_set))}
    check("Full datasets",
          len(train_set) == EXPECTED_SIZES["train"] and len(valid_set) == EXPECTED_SIZES["valid"]
          and distribution == {"train": {"FAKE": 50_000, "REAL": 50_000}, "valid": {"FAKE": 10_000, "REAL": 10_000}},
          f"train {len(train_set):,} {distribution['train']}, valid {len(valid_set):,} {distribution['valid']}")
    check("Validation transform deterministic", not random_operations(valid_test_transform),
          f"validation: no random ops; training: {random_operations(train_transform)}")

    train_loader, valid_loader, generator = build_loaders(train_set, valid_set, config, use_cuda)
    generator.manual_seed(config["seed"] + 1)
    images, labels = next(iter(train_loader))     # starts 4 workers -> each self-tests its guard
    valid_images, valid_labels = next(iter(valid_loader))
    check("Training loader",
          isinstance(train_loader.sampler, RandomSampler) and len(train_loader) == 12_500
          and tuple(images.shape) == (8, 3, 224, 224) and labels.dtype == torch.int64
          and train_loader.num_workers == 4 and not train_loader.persistent_workers
          and train_loader.worker_init_fn is guard_worker,
          "12,500 shuffled batches of 8, 4 workers (re-seeded each epoch), test guard in every worker")
    check("Validation loader",
          isinstance(valid_loader.sampler, SequentialSampler) and len(valid_loader) == 625
          and tuple(valid_images.shape) == (32, 3, 224, 224) and valid_loader.num_workers == 4
          and valid_loader.worker_init_fn is guard_worker,
          "625 batches of 32 in fixed order, 4 workers, test guard in every worker")

    # 3. Model
    print("\nModel")
    model, stage1_checkpoint = build_model_from_stage1(config, device)
    from_stage1 = all(torch.equal(t.cpu(), stage1_checkpoint["model_state_dict"][k])
                      for k, t in model.state_dict().items())
    trainable, frozen = count_parameters(model)
    check("Initialised from Stage 1 best.pt", from_stage1,
          f"all {len(stage1_checkpoint['model_state_dict'])} tensors equal Stage 1 epoch "
          f"{stage1_checkpoint['epoch']} (val loss {stage1_checkpoint['metrics']['val_loss']:.4f}), not random")
    check("Trainable parameters", trainable == EXPECTED_TRAINABLE and frozen == 0,
          f"{trainable:,} trainable, {frozen:,} frozen")
    set_train_mode(model, freeze_batchnorm=config["freeze_batchnorm"])
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    check("BatchNorm statistics frozen", all(not bn.training for bn in bn_layers) and model.swin.training,
          f"{len(bn_layers)} BatchNorm layers in eval mode while the rest trains")

    # 4. Optimizer / scheduler / AMP
    print("\nOptimizer / scheduler / AMP")
    optimizer, scheduler = build_optimizer(model, config)
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
    check("Optimizer", type(optimizer).__name__ == "AdamW"
          and in_optimizer == {id(p) for p in model.parameters()}
          and optimizer.param_groups[0]["lr"] == 1e-5 and optimizer.param_groups[0]["weight_decay"] == 0.01,
          f"AdamW over all {len(in_optimizer)} parameter tensors, lr 1e-5, weight decay 0.01")
    preview_opt, preview_sched = build_optimizer(nn.Linear(1, 1), config)
    lrs = []
    for _ in range(config["epochs"]):
        lrs.append(preview_opt.param_groups[0]["lr"])
        preview_opt.step()
        preview_sched.step()
    check("Scheduler", type(scheduler).__name__ == "CosineAnnealingLR" and lrs == sorted(lrs, reverse=True),
          "per-epoch learning rates " + ", ".join(f"{lr:.2e}" for lr in lrs))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        amp_logits = model(images[:2].to(device))
    check("AMP", scaler.is_enabled() and amp_logits.dtype == torch.float16,
          "torch.amp.autocast('cuda') fp16 + torch.amp.GradScaler('cuda'); no deprecated API")

    # 5. Real training steps on the real 100k loader
    print(f"\nReal training steps (batch 8, gradient clipping {config['max_grad_norm']})")
    criterion = nn.CrossEntropyLoss()
    parts = {"Swin": model.swin, "EfficientNet": model.efficientnet,
             "Cross-attention": model.fusion, "Classifier": model.classifier}
    before = {name: snapshot(m) for name, m in parts.items()}
    bn_before = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in bn_layers]
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                        max_batches=DRY_WARMUP_STEPS, max_grad_norm=config["max_grad_norm"],
                        freeze_batchnorm=config["freeze_batchnorm"])
        # Same work as train_one_epoch, but each step is timed (data wait included)
        set_train_mode(model, freeze_batchnorm=config["freeze_batchnorm"])
        timed = []
        last = time.perf_counter()
        for step, (batch_images, batch_labels) in enumerate(train_loader, start=1):
            result = train_step(model, batch_images, batch_labels, criterion, optimizer, scaler,
                                device, use_amp, max_grad_norm=config["max_grad_norm"])
            now = time.perf_counter()
            timed.append({**result, "seconds": now - last})
            last = now
            if step >= DRY_TIMED_STEPS:
                break
    nondeterministic = sorted({str(w.message).split(" does not")[0] for w in caught
                               if "deterministic implementation" in str(w.message)})
    dry_losses = [r["loss"] for r in timed]
    applied = [r for r in timed if r["step_applied"]]
    changes = {name: changed(m, before[name]) for name, m in parts.items()}
    bn_fixed = all(torch.equal(bn.running_mean, m) and torch.equal(bn.running_var, v)
                   for bn, (m, v) in zip(bn_layers, bn_before))
    check("Forward / backward / optimizer step",
          all(math.isfinite(x) for x in dry_losses) and applied and all(r["grads_finite"] for r in applied),
          f"{DRY_WARMUP_STEPS + DRY_TIMED_STEPS} steps; mean loss {sum(dry_losses) / len(dry_losses):.4f}, "
          f"no NaN/Inf; {len(timed) - len(applied)} fp16-overflow steps skipped by GradScaler")
    check("Weights updated in all four parts", all(changes.values()) and bn_fixed,
          ", ".join(f"{k} {v} tensors" for k, v in changes.items()) + "; BatchNorm statistics unchanged")

    # 6. Validation on a small deterministic, balanced subset
    print("\nValidation (small deterministic subset)")
    valid_subset = balanced_subset(valid_set, DRY_VALID_PER_CLASS, config["seed"])
    subset_loader = build_loader(valid_subset, config["eval_batch_size"], False, config["num_workers"], use_cuda)
    evaluate(model, subset_loader, criterion, device, use_amp)  # starts the persistent workers
    t3 = time.perf_counter()
    valid_stats = evaluate(model, subset_loader, criterion, device, use_amp)
    t4 = time.perf_counter()
    m = valid_stats["class_metrics"]
    check("Validation", math.isfinite(valid_stats["loss"]) and valid_stats["total"] == 2 * DRY_VALID_PER_CLASS
          and [sum(r) for r in valid_stats["confusion"]] == [DRY_VALID_PER_CLASS] * 2,
          f"{valid_stats['total']} images {class_counts(valid_subset)}, loss {valid_stats['loss']:.4f}, "
          f"acc {valid_stats['accuracy']:.1%}, FAKE F1 {m['FAKE']['f1']:.3f}, REAL F1 {m['REAL']['f1']:.3f}")

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if use_cuda else 0
    check("GPU memory", use_cuda and peak_reserved < free_mb,
          f"peak {peak_alloc:.0f} MB allocated / {peak_reserved:.0f} MB reserved of {free_mb:.0f} MB free")

    # 7. Temporary checkpoint + resume round trip (deleted afterwards)
    print("\nCheckpoint / resume")
    try:
        dry_train_stats = {"loss": sum(dry_losses) / len(dry_losses),
                           "accuracy": sum(r["correct"] for r in timed) / (8 * len(timed)),
                           "steps_skipped": len(timed) - len(applied)}
        record = epoch_record(1, lrs[0], dry_train_stats, valid_stats,
                              sum(r["seconds"] for r in timed), t4 - t3, peak_alloc)
        record["is_best"] = True
        scheduler.step()
        save_stage2_checkpoint(DRY_RUN_CHECKPOINT, model, optimizer, scheduler, scaler, 1, config,
                               [record], valid_stats["loss"], 1)
        size_mb = DRY_RUN_CHECKPOINT.stat().st_size / 1024**2
        fresh = empty_model(config, device)           # random weights: all must come from the file
        fresh_opt, fresh_sched = build_optimizer(fresh, config)
        fresh_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        restored = restore_stage2(DRY_RUN_CHECKPOINT, fresh, fresh_opt, fresh_sched, fresh_scaler, device, config)
        model.eval(); fresh.eval()
        with torch.no_grad():
            identical = torch.equal(model(valid_images[:8].to(device)), fresh(valid_images[:8].to(device)))
        state_ok = (fresh_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
                    and fresh_sched.last_epoch == scheduler.last_epoch
                    and fresh_scaler.get_scale() == scaler.get_scale()
                    and restored["best_val_loss"] == valid_stats["loss"] and restored["epoch"] == 1)
        check("Checkpoint save / resume", identical and state_ok,
              f"temporary {DRY_RUN_CHECKPOINT.relative_to(PROJECT_ROOT)} ({size_mb:.0f} MB): model, optimizer, "
              f"scheduler, scaler, epoch, best loss and history restored into a random-weight model; "
              f"resume would start at epoch {restored['epoch'] + 1}")
    except Exception as exc:
        check("Checkpoint save / resume", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(DRY_RUN_DIR, ignore_errors=True)

    # 8. Final safety
    print("\nFinal safety")
    created = [p.name for p in (LAST_PATH, BEST_PATH, HISTORY_PATH, SUMMARY_PATH) if p.exists()]
    check("No final Stage 2 outputs created", not created and not DRY_RUN_DIR.exists(),
          "no last.pt / best.pt / history.csv; temporary dry-run checkpoint removed"
          if not created else f"found {created}")
    stage1_ok = fingerprint_stage1() == stage1_before
    check("Stage 1 artifacts unchanged", stage1_ok, f"{len(stage1_before)} files, same SHA-256")
    guard.active = False
    check("Test data not accessed", not guard.blocked,
          "0 accesses in the main process; every worker ran with an active guard")

    # Reproducibility record (no timings, no timestamps)
    steady = sorted(r["seconds"] for r in timed[DRY_STARTUP_STEPS:])
    train_per_step = steady[len(steady) // 2]   # median: robust to start-up and hiccups
    step_p90 = steady[int(len(steady) * 0.9)]
    valid_per_batch = (t4 - t3) / len(subset_loader)
    facts = {
        "train_images": len(train_set), "valid_images": len(valid_set), "distribution": distribution,
        "trainable": trainable, "frozen": frozen, "config": config, "lr_per_epoch": lrs,
        "first_train_batch_labels": labels.tolist(),
        "valid_subset_sha256": hashlib.sha256(str(valid_subset.indices).encode()).hexdigest(),
        "dry_losses": dry_losses, "dry_valid": {"loss": valid_stats["loss"],
                                                "confusion": valid_stats["confusion"]},
        "weights_after_dry_steps_sha256": weights_fingerprint(model),
        "nondeterministic_ops": nondeterministic,
    }
    write_json(DRY_SUMMARY_PATH, facts)

    print("\n" + "=" * 64)
    print("DRY RUN REPORT")
    print("=" * 64)
    rows = [
        ("1. Training images", f"{len(train_set):,}"),
        ("2. Validation images", f"{len(valid_set):,}"),
        ("3. Class distribution", f"train {distribution['train']}, valid {distribution['valid']}"),
        ("4. Trainable parameters", f"{trainable:,}"),
        ("5. Frozen parameters", f"{frozen:,}"),
        ("6. Batch size", f"{config['batch_size']} (validation {config['eval_batch_size']})"),
        ("7. DataLoader workers", f"{config['num_workers']} (train re-seeded per epoch; guard in each)"),
        ("8. Learning rate", f"{config['learning_rate']:.0e} -> cosine to {config['min_learning_rate']:.0e}"),
        ("9. Optimizer", f"AdamW, weight decay {config['weight_decay']}"),
        ("10. Scheduler", f"CosineAnnealingLR over {config['epochs']} epochs"),
        ("11. AMP", f"{'enabled' if use_amp else 'disabled'} (fp16 autocast + GradScaler)"),
        ("12. Gradient clipping", f"max norm {config['max_grad_norm']}"),
        ("13. GPU memory", f"{peak_alloc:.0f} MB allocated / {peak_reserved:.0f} MB reserved (of {free_mb:.0f} free)"),
        ("14. Checkpoint paths", f"{LAST_PATH.relative_to(PROJECT_ROOT)}, {BEST_PATH.relative_to(PROJECT_ROOT)}"),
        ("15. Test data accessed", "NO" if not guard.blocked else f"YES ({len(guard.blocked)})"),
        ("16. Stage 1 artifacts modified", "NO" if stage1_ok else "YES"),
    ]
    for label, value in rows:
        print(f"{label:<32}: {value}")
    epoch_min = (train_per_step * len(train_loader) + valid_per_batch * len(valid_loader)) / 60
    print(f"\nMeasured speed: median {train_per_step:.3f} s/step, 90th percentile {step_p90:.3f} s/step "
          f"(steps {DRY_STARTUP_STEPS + 1}-{DRY_TIMED_STEPS}, 4 workers, data wait included)")
    print(f"Estimate: ~{epoch_min:.0f} min/epoch, ~{epoch_min * config['epochs'] / 60:.1f} h for "
          f"{config['epochs']} epochs (first step of each epoch adds worker start-up)")
    print(f"Weights fingerprint after dry steps: {facts['weights_after_dry_steps_sha256'][:16]}")
    print(f"Non-deterministic ops reported: {nondeterministic or 'none'}")
    print()
    for name, passed in results.items():
        print(f"{name:<38}: {'PASS' if passed else 'FAIL'}")
    ready = all(results.values())
    print("\n" + "=" * 64)
    print("READY" if ready else "NOT READY")
    print("=" * 64)
    return ready


def class_counts_full(dataset):
    return {CLASS_NAMES[c]: dataset.targets.count(c) for c in CLASS_NAMES}


def main():
    parser = argparse.ArgumentParser(description="DeFaX Stage 2 fine-tuning (all parameters trainable)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="run every check on a few batches; no full training")
    mode.add_argument("--resume", action="store_true", help="continue Stage 2 from models/checkpoints/stage2/last.pt")
    args = parser.parse_args()
    config = dict(STAGE2_CONFIG)

    if args.dry_run:
        return dry_run(config)
    try:
        return train(config, resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Continue with:  python -m src.training.train_stage2 --resume "
              "(restarts the interrupted epoch from the last completed one)")
        return False


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    sys.exit(0 if main() else 1)
