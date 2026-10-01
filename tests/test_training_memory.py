"""
GPU memory feasibility benchmark for ONE training step of the DeFaX model.

This is NOT training and says nothing about accuracy:
- random weights (pretrained=False), nothing downloaded
- synthetic random images, the real dataset is never opened
- a fresh model and optimizer for every configuration, deleted afterwards

For each configuration it runs: forward -> CrossEntropyLoss -> backward ->
AdamW step, and records the peak CUDA memory.

Note for Windows: recent NVIDIA drivers can spill into system RAM instead of
raising out-of-memory. Such a step "succeeds" but would make real training
very slow, so peaks above the free VRAM are flagged separately.
"""

import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

REQUESTED_BATCHES = [1, 2, 4, 8]
EXTRA_BATCHES = [16, 32]          # extra probe: the data pipeline uses 32
MAX_AMP_ATTEMPTS = 10             # GradScaler may skip the first steps (see below)

FAKE, REAL = 0, 1


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def one_training_step(batch_size, use_amp, freeze_backbones, budget_mb):
    """Run forward + loss + backward + optimizer step. Returns a result dict."""
    from src.models.defax import DeFaX

    device = torch.device("cuda")
    model = optimizer = images = labels = logits = loss = scaler = None
    clear_cuda()
    torch.cuda.reset_peak_memory_stats()

    try:
        torch.manual_seed(0)
        model = DeFaX(pretrained=False, freeze_backbones=freeze_backbones).to(device)
        model.train()  # dropout/BatchNorm in training mode, as in real training

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        images = torch.randn(batch_size, 3, 224, 224, device=device)
        labels = (torch.arange(batch_size, device=device) % 2)  # FAKE, REAL, FAKE, ...

        watched = model.classifier.layers[-1].weight
        before = watched.detach().clone()

        # With fp16 AMP the GradScaler starts with a very large loss scale and
        # skips the optimizer step if gradients overflow. AdamW only allocates
        # its memory on the first REAL step, so repeat until a step is applied;
        # otherwise the optimizer memory would be missing from the measurement.
        attempts = 0
        start = time.perf_counter()
        while attempts < (MAX_AMP_ATTEMPTS if use_amp else 1):
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not torch.equal(watched.detach(), before):
                break
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        step_applied = not torch.equal(watched.detach(), before)
        finite = bool(torch.isfinite(loss).item())
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

        if not (step_applied and finite):
            status = "FAIL"
        elif peak_reserved > budget_mb:
            status = "SPILL"  # worked only by overflowing into system RAM
        else:
            status = "PASS"

        return {
            "status": status,
            "peak_alloc": peak_alloc,
            "peak_reserved": peak_reserved,
            "seconds": seconds,
            "attempts": attempts,
            "step_applied": step_applied,
            "trainable": sum(p.numel() for p in trainable),
        }

    except torch.OutOfMemoryError:
        return {"status": "OOM", "peak_alloc": torch.cuda.max_memory_allocated() / 1024**2}
    except Exception as error:
        return {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    finally:
        del model, optimizer, images, labels, logits, loss, scaler
        clear_cuda()


def run_section(title, use_amp, freeze_backbones, budget_mb):
    print(f"\n{title}:")
    results = {}
    oom_seen = False
    for batch in REQUESTED_BATCHES + EXTRA_BATCHES:
        extra = "  (extra)" if batch in EXTRA_BATCHES else ""
        if oom_seen:
            results[batch] = {"status": "SKIPPED"}
            print(f"  Batch {batch:>2}: SKIPPED - a smaller batch already ran out of memory{extra}")
            continue

        r = one_training_step(batch, use_amp, freeze_backbones, budget_mb)
        results[batch] = r
        if r["status"] == "OOM":
            oom_seen = True
            print(f"  Batch {batch:>2}: OOM  - out of memory{extra}")
        elif "error" in r:
            print(f"  Batch {batch:>2}: FAIL - {r['error']}{extra}")
        else:
            note = f", {r['attempts']} scaler attempts" if r["attempts"] > 1 else ""
            label = "PASS" if r["status"] == "PASS" else r["status"]
            print(
                f"  Batch {batch:>2}: {label:<5}- peak {r['peak_alloc']:>6.0f} MB allocated "
                f"/ {r['peak_reserved']:>6.0f} MB reserved, {r['seconds']:.2f}s{note}{extra}"
            )
    return results


def largest_safe(results):
    safe = [b for b, r in results.items() if r["status"] == "PASS"]
    return max(safe) if safe else None


def main():
    print("=" * 60)
    print("TRAINING MEMORY FEASIBILITY")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available - this benchmark needs the GPU.")
        return False

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    budget_mb = free_bytes / 1024**2
    print(f"\nGPU:\n  {torch.cuda.get_device_name(0)}")
    print(f"\nVRAM:\n  {total_bytes / 1024**2:.0f} MB total, {budget_mb:.0f} MB free at start "
          f"(the rest is used by Windows and open apps)")
    print("\nWeights: random | Data: synthetic | Optimizer: AdamW (lr 1e-4) | Loss: CrossEntropyLoss")
    print("Each configuration uses a fresh model; 'SPILL' = only fits by using system RAM (very slow)")

    fp32 = run_section("A. FP32 (full model trainable)", use_amp=False,
                       freeze_backbones=False, budget_mb=budget_mb)
    amp = run_section("B. Mixed precision, fp16 AMP (full model trainable)", use_amp=True,
                      freeze_backbones=False, budget_mb=budget_mb)
    frozen = run_section("C. Extra: mixed precision + frozen backbones", use_amp=True,
                         freeze_backbones=True, budget_mb=budget_mb)

    sections = {"FP32": fp32, "AMP": amp, "AMP + frozen": frozen}
    all_runs = [r for s in sections.values() for r in s.values() if "step_applied" in r]
    optimizer_ok = bool(all_runs) and all(r["step_applied"] for r in all_runs)
    print(f"\nOptimizer step:\n  {'PASS' if optimizer_ok else 'FAIL'}"
          f" (weights changed after every completed step)")

    trainable_full = next((r["trainable"] for r in fp32.values() if "trainable" in r), None)
    trainable_frozen = next((r["trainable"] for r in frozen.values() if "trainable" in r), None)
    if trainable_full and trainable_frozen:
        print(f"\nTrainable parameters:\n  full model {trainable_full:,} | "
              f"frozen backbones {trainable_frozen:,}")

    print("\nLargest batch that fits in free VRAM (no spill):")
    for name, results in sections.items():
        print(f"  {name:<13}: {largest_safe(results) or 'none'}")

    # A requested batch size failing counts as a failure of the benchmark;
    # OOM/SPILL on the extra sizes is information, not failure
    requested_ok = all(
        sections[s][b]["status"] in ("PASS", "SPILL", "OOM", "SKIPPED")
        and "error" not in sections[s][b]
        for s in sections for b in REQUESTED_BATCHES
    )
    passed = requested_ok and optimizer_ok
    print(f"\nBenchmark result:\n  {'PASS' if passed else 'FAIL'} (completed without crashing)")
    print("\nThis measures memory only. It says nothing about accuracy.")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
