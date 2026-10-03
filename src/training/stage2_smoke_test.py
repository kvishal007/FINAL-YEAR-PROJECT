"""
LEVEL 4.5 - Stage 2 fine-tuning SMOKE TEST (not real training).

Full DeFaX (Swin-T + EfficientNet-B0 UNFROZEN), initialised from the Stage 1
best checkpoint, trained for 2 short epochs on a fixed 400-image training
subset and validated on a fixed 200-image validation subset.

The test split is locked: TestSplitGuard blocks any access to it for the
whole run, and num_workers=0 keeps all image loading inside this process
where the guard can see it.

Run from the project root:
    python -m src.training.stage2_smoke_test
"""

import hashlib
import json
import math
import os
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

from src.evaluation.evaluate_test import sha256
from src.training.smoke_test import balanced_subset, class_counts, labels_match_folders
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
    random_operations,
    save_stage2_checkpoint,
    snapshot,
    stage1_artifact_files,
    weights_fingerprint,
)
from src.training.train_stage1 import git_ignored
from src.training.trainer import (
    CLASS_NAMES,
    count_parameters,
    evaluate,
    load_checkpoint,
    set_train_mode,
    train_one_epoch,
)
from src.data.transforms import train_transform, valid_test_transform

TRAIN_PER_CLASS = 200     # 400 training images  -> 50 steps per epoch at batch 8
VALID_PER_CLASS = 100     # 200 validation images
EPOCHS = 2
CHECKPOINT_PATH = STAGE2_CHECKPOINT_DIR / "smoke_test.pt"
SUMMARY_PATH = STAGE2_RESULTS_DIR / "smoke_test_summary.json"
STAGE1_FILES = stage1_artifact_files()

checks = {}


def check(name, passed, note=""):
    checks[name] = bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({note})" if note else ""), flush=True)
    return bool(passed)


def main():
    config = dict(STAGE2_CONFIG)
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    nondeterministic_ops = set()
    warnings.filterwarnings("error", category=FutureWarning)  # no deprecated APIs (e.g. old AMP)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = config["amp"] and use_cuda
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2 if use_cuda else None
    start = time.perf_counter()

    print("=" * 64)
    print("LEVEL 4.5 - STAGE 2 FINE-TUNING SMOKE TEST")
    print("=" * 64)

    # ---------------- Safety ----------------
    print("\nSafety")
    guard = TestSplitGuard.install()
    guard.active = True
    try:
        os.listdir(TEST_DIR)
        guard_works = False
    except PermissionError:
        guard_works = True
    self_test_blocks = len(guard.blocked)
    guard.blocked.clear()
    check("Test-split guard active", guard_works,
          "a deliberate listing of test/ was blocked; guard stays on for the whole run")

    refused = []
    for target in [*PROTECTED_PATHS, PROJECT_ROOT / "data" / "x.pt", PROJECT_ROOT / "models" / "checkpoints" / "x.pt"]:
        try:
            assert_stage2_output(target)
        except PermissionError:
            refused.append(target)
    allowed = assert_stage2_output(CHECKPOINT_PATH) and assert_stage2_output(SUMMARY_PATH)
    check("Stage 1 outputs write-protected",
          len(refused) == len(PROTECTED_PATHS) + 2 and allowed,
          f"refused all {len(PROTECTED_PATHS)} Stage 1 paths + data/ + checkpoints root; Stage 2 paths allowed")

    stage1_before = {p: sha256(p) for p in STAGE1_FILES}
    stage2_ignored = all(git_ignored(p) for p in (CHECKPOINT_PATH, SUMMARY_PATH,
                                                  STAGE2_CHECKPOINT_DIR / "best.pt",
                                                  STAGE2_CHECKPOINT_DIR / "last.pt"))
    check("Stage 2 paths separate and git-ignored", stage2_ignored,
          f"{STAGE2_CHECKPOINT_DIR.relative_to(PROJECT_ROOT)}, {STAGE2_RESULTS_DIR.relative_to(PROJECT_ROOT)}")

    # ---------------- Data ----------------
    print("\nData")
    train_full, valid_full = build_datasets()  # train/ and valid/ only; checks split membership
    train_set = balanced_subset(train_full, TRAIN_PER_CLASS, seed)
    valid_set = balanced_subset(valid_full, VALID_PER_CLASS, seed)
    repeatable = balanced_subset(train_full, TRAIN_PER_CLASS, seed).indices == train_set.indices
    check("Train/validation data only",
          labels_match_folders(train_set) and labels_match_folders(valid_set) and repeatable,
          f"train {len(train_set)} {class_counts(train_set)}, valid {len(valid_set)} {class_counts(valid_set)}, "
          "subset repeatable, labels match folders")
    check("No random augmentation on validation",
          not random_operations(valid_test_transform),
          f"validation: none; training: {random_operations(train_transform)}")

    generator = torch.Generator().manual_seed(seed)
    train_loader = build_loader(train_set, config["batch_size"], True, 0, use_cuda, generator)
    valid_loader = build_loader(valid_set, config["eval_batch_size"], False, 0, use_cuda)

    # ---------------- Model ----------------
    print("\nModel")
    stage1_sha = sha256(STAGE1_BEST)
    model, stage1_checkpoint = build_model_from_stage1(config, device)
    trainable, frozen = count_parameters(model)
    parts = {"Swin": model.swin, "EfficientNet": model.efficientnet,
             "Cross-attention": model.fusion, "Classifier": model.classifier}
    all_trainable = {name: all(p.requires_grad for p in m.parameters()) for name, m in parts.items()}
    check("All four parts trainable", all(all_trainable.values()) and frozen == 0,
          f"trainable {trainable:,}, frozen {frozen:,}")

    set_train_mode(model, freeze_batchnorm=config["freeze_batchnorm"])
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    sd_layers = [m for m in model.modules() if "StochasticDepth" in type(m).__name__]
    check("Train-mode behaviour",
          model.swin.training and all(not bn.training for bn in bn_layers)
          and all(m.training for m in sd_layers),
          f"{len(bn_layers)} BatchNorm layers keep fixed statistics; {len(sd_layers)} stochastic-depth layers active")

    criterion = nn.CrossEntropyLoss()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        baseline = evaluate(model, valid_loader, criterion, device, use_amp)
    check("Stage 1 weights loaded", baseline["accuracy"] > 0.9,
          f"epoch {stage1_checkpoint['epoch']} model on validation subset before Stage 2: "
          f"loss {baseline['loss']:.4f}, acc {baseline['accuracy']:.1%} (random weights would give ~50%)")

    images, labels = next(iter(train_loader))
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(images.to(device))
    check("Forward pass", tuple(logits.shape) == (config["batch_size"], 2) and images.shape[0] == 8,
          f"batch {list(images.shape)} -> logits {list(logits.shape)}")

    # ---------------- Training ----------------
    print(f"\nTraining ({EPOCHS} epochs x {len(train_loader)} steps, batch {config['batch_size']}, "
          f"lr {config['learning_rate']:.0e}, AMP {use_amp})")
    optimizer, scheduler = build_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    before = {name: snapshot(m) for name, m in parts.items()}
    bn_stats_before = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in bn_layers]
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    history, epoch_times = [], []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            for epoch in range(1, EPOCHS + 1):
                generator.manual_seed(seed + epoch)
                lr = optimizer.param_groups[0]["lr"]
                t0 = time.perf_counter()
                train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                              use_amp, max_grad_norm=config["max_grad_norm"],
                                              freeze_batchnorm=config["freeze_batchnorm"])
                t1 = time.perf_counter()
                valid_stats = evaluate(model, valid_loader, criterion, device, use_amp)
                t2 = time.perf_counter()
                if scheduler:
                    scheduler.step()
                epoch_times.append((t1 - t0, t2 - t1))
                history.append({"epoch": epoch, "learning_rate": lr,
                                "train_loss": train_stats["loss"], "train_accuracy": train_stats["accuracy"],
                                "val_loss": valid_stats["loss"], "val_accuracy": valid_stats["accuracy"],
                                "fake_f1": valid_stats["class_metrics"]["FAKE"]["f1"],
                                "real_f1": valid_stats["class_metrics"]["REAL"]["f1"],
                                "steps_applied": train_stats["steps_applied"],
                                "steps_skipped": train_stats["steps_skipped"],
                                "nonfinite": train_stats["applied_with_nonfinite_grads"]})
                print(f"    epoch {epoch}: lr {lr:.2e} | train loss {train_stats['loss']:.4f} "
                      f"acc {train_stats['accuracy']:.1%} | valid loss {valid_stats['loss']:.4f} "
                      f"acc {valid_stats['accuracy']:.1%} | {t1 - t0:.1f}s train, {t2 - t1:.1f}s valid", flush=True)
        for w in caught:
            if "does not have a deterministic implementation" in str(w.message):
                nondeterministic_ops.add(str(w.message).split(" does not")[0])
        error = None
    except torch.OutOfMemoryError as exc:
        error = f"CUDA out of memory: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if error:
        print(f"  Training failed: {error}")
        check("Training", False, error)
        return report(config, trainable, frozen, None, None, None, use_cuda, free_mb)

    total_steps = sum(h["steps_applied"] + h["steps_skipped"] for h in history)
    check("Forward + backward + optimizer step",
          total_steps == EPOCHS * len(train_loader) and sum(h["steps_applied"] for h in history) > 0,
          f"{total_steps} steps, {sum(h['steps_applied'] for h in history)} applied, "
          f"{sum(h['steps_skipped'] for h in history)} skipped by GradScaler")
    check("No NaN/Inf in loss or gradients",
          all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) and h["nonfinite"] == 0
              for h in history))

    changes = {name: (changed(m, before[name]), len(before[name])) for name, m in parts.items()}
    check("Weights changed in every part",
          all(c > 0 for c, _ in changes.values()),
          ", ".join(f"{name} {c}/{n}" for name, (c, n) in changes.items()) + " tensors updated")
    bn_fixed = all(torch.equal(bn.running_mean, m) and torch.equal(bn.running_var, v)
                   for bn, (m, v) in zip(bn_layers, bn_stats_before))
    check("BatchNorm statistics fixed", bn_fixed, "running mean/var unchanged; BN weights still trained")

    final = history[-1]
    check("Validation", math.isfinite(final["val_loss"]) and 0 <= final["val_accuracy"] <= 1,
          f"loss {final['val_loss']:.4f}, acc {final['val_accuracy']:.1%}, "
          f"FAKE F1 {final['fake_f1']:.3f}, REAL F1 {final['real_f1']:.3f}")

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if use_cuda else None
    check("GPU memory", use_cuda and peak_reserved < free_mb,
          f"peak {peak_alloc:.0f} MB allocated / {peak_reserved:.0f} MB reserved, "
          f"{free_mb:.0f} MB free at start" if use_cuda else "no CUDA")

    # ---------------- Checkpoint ----------------
    print("\nCheckpoint")
    try:
        save_stage2_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, scaler, EPOCHS, config,
                               history, final["val_loss"], EPOCHS)
        fresh, _ = build_model_from_stage1(config, device)   # starts as Stage 1 weights
        fresh_opt, fresh_sched = build_optimizer(fresh, config)
        fresh_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        saved = load_checkpoint(CHECKPOINT_PATH, fresh, fresh_opt, fresh_scaler, device=device)
        fresh_sched.load_state_dict(saved["scheduler_state_dict"])
        model.eval(); fresh.eval()
        with torch.no_grad():
            same = torch.equal(model(images.to(device)), fresh(images.to(device)))
        state_ok = (fresh_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
                    and fresh_sched.last_epoch == scheduler.last_epoch
                    and fresh_scaler.get_scale() == scaler.get_scale()
                    and saved["init_checkpoint"] == config["init_checkpoint"])
        check("Checkpoint save / reload", same and state_ok,
              f"{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)} "
              f"({CHECKPOINT_PATH.stat().st_size / 1024**2:.0f} MB); identical outputs; "
              "optimizer, scheduler and scaler restored")
    except Exception as exc:
        check("Checkpoint save / reload", False, f"{type(exc).__name__}: {exc}")

    # ---------------- Final safety ----------------
    print("\nFinal safety")
    stage1_after = {p: sha256(p) for p in STAGE1_FILES}
    check("Stage 1 files untouched", stage1_before == stage1_after and sha256(STAGE1_BEST) == stage1_sha,
          f"{len(STAGE1_FILES)} files incl. best.pt, last.pt, stage1 history/test results: same SHA-256")
    guard.active = False
    check("Test dataset never accessed", not guard.blocked,
          f"0 test-split accesses during the run (guard self-test blocked {self_test_blocks} on purpose)")

    fingerprint = weights_fingerprint(model)
    summary = {
        "config": config,
        "train_indices_sha256": hashlib.sha256(str(train_set.indices).encode()).hexdigest(),
        "valid_indices_sha256": hashlib.sha256(str(valid_set.indices).encode()).hexdigest(),
        "baseline_val_loss": baseline["loss"],
        "history": history,
        "final_weights_sha256": fingerprint,
        "nondeterministic_ops": sorted(nondeterministic_ops),
    }
    assert_stage2_output(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    train_s_per_step = epoch_times[-1][0] / len(train_loader)   # epoch 2: no warm-up cost
    valid_s_per_batch = epoch_times[-1][1] / len(valid_loader)
    estimate = {
        "train_s_per_step": train_s_per_step,
        "epoch_min": (train_s_per_step * 100_000 / config["batch_size"]
                      + valid_s_per_batch * 20_000 / config["eval_batch_size"]) / 60,
    }
    print(f"\nTotal time: {time.perf_counter() - start:.0f}s")
    return report(config, trainable, frozen, (peak_alloc, peak_reserved), history, (baseline, fingerprint,
                  nondeterministic_ops, estimate), use_cuda, free_mb)


def report(config, trainable, frozen, memory, history, extra, use_cuda, free_mb):
    print("\n" + "=" * 64)
    print("STAGE 2 SMOKE TEST SUMMARY")
    print("=" * 64)
    print(f"Device              : {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")
    print(f"Batch size          : {config['batch_size']} (validation {config['eval_batch_size']})")
    print(f"Learning rate       : {config['learning_rate']:.0e} AdamW, wd {config['weight_decay']}, "
          f"{config['scheduler']} schedule, grad clip {config['max_grad_norm']}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {frozen:,}")
    if memory:
        print(f"Peak GPU memory     : {memory[0]:.0f} MB allocated / {memory[1]:.0f} MB reserved "
              f"(of {free_mb:.0f} MB free)")
    if history and extra:
        baseline, fingerprint, nondeterministic, estimate = extra
        print(f"Validation (subset) : before {baseline['accuracy']:.1%} / loss {baseline['loss']:.4f} -> "
              f"after {history[-1]['val_accuracy']:.1%} / loss {history[-1]['val_loss']:.4f}")
        print(f"Weights fingerprint : {fingerprint[:16]}")
        print(f"Non-deterministic ops reported by PyTorch: {sorted(nondeterministic) or 'none'}")
        print(f"Speed estimate      : {estimate['train_s_per_step']:.3f} s/step -> ~{estimate['epoch_min']:.0f} min "
              f"per full epoch (12,500 steps + validation; num_workers=0 here)")
    print()
    for name, passed in checks.items():
        print(f"{name:<40}: {'PASS' if passed else 'FAIL'}")
    overall = bool(checks) and all(checks.values()) and history is not None
    print("\n" + "=" * 64)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 64)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
