"""
Read-only test for src/models/classifier.py.

Uses a random dummy fused vector [2, 256]. No training: the module runs in
eval mode inside torch.no_grad().
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

BATCH = 2
INPUT_DIM = 256
NUM_CLASSES = 2
INPUT_SHAPE = (BATCH, INPUT_DIM)

results = {}


def record(name, passed, note=""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({note})" if note else ""))


def rejects(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def run_forward(model, device):
    model = model.to(device)
    torch.manual_seed(0)
    fused = torch.randn(INPUT_SHAPE).to(device)
    with torch.no_grad():
        logits = model(fused)
    if logits.device.type != device.type:
        raise RuntimeError(f"Output on {logits.device}, expected {device}")
    return logits.cpu()


def main():
    print("=" * 50)
    print("CLASSIFIER TEST")
    print("=" * 50)

    # 1. Import
    try:
        from src.models.classifier import CLASS_NAMES, Classifier
        record("Import", True)
    except Exception as error:
        record("Import", False, f"{type(error).__name__}: {error}")
        return False

    # 2. Initialization
    try:
        model = Classifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).eval()
        n_params = sum(p.numel() for p in model.parameters())
        mapping_ok = CLASS_NAMES == {0: "FAKE", 1: "REAL"}
        record("Initialization", mapping_ok,
               f"{n_params:,} parameters, labels {CLASS_NAMES}")
    except Exception as error:
        record("Initialization", False, f"{type(error).__name__}: {error}")
        return False

    # 3. Input shape: the expected [2, 256] is accepted
    try:
        with torch.no_grad():
            model(torch.zeros(INPUT_SHAPE))
        record("Input shape", True, f"accepts {list(INPUT_SHAPE)}")
    except Exception as error:
        record("Input shape", False, f"{type(error).__name__}: {error}")

    # 4. Forward pass (CPU)
    cpu = torch.device("cpu")
    try:
        logits = run_forward(model, cpu)
        record("Forward pass", True)
    except Exception as error:
        record("Forward pass", False, f"{type(error).__name__}: {error}")
        return False

    # 5. Output shape
    record("Output shape [B,2]", tuple(logits.shape) == (BATCH, NUM_CLASSES), str(list(logits.shape)))

    # 6. Float output (raw logits; softmax turns them into probabilities)
    probs = logits.softmax(dim=1)
    float_ok = logits.dtype == torch.float32 and torch.allclose(probs.sum(dim=1), torch.ones(BATCH))
    record("Float output", float_ok, f"{logits.dtype}, softmax rows sum to 1")

    # 7. CUDA
    gpu_logits = None
    peak_mb = None
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            gpu_logits = run_forward(model, torch.device("cuda"))
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            record("CUDA", torch.allclose(gpu_logits, logits, atol=1e-4),
                   f"{torch.cuda.get_device_name(0)}, matches CPU result")
        except Exception as error:
            record("CUDA", False, f"{type(error).__name__}: {error}")
        finally:
            model = model.to(cpu)
            torch.cuda.empty_cache()
    else:
        record("CUDA", False, "CUDA not available on this machine")

    # 8. CPU fallback
    try:
        record("CPU fallback", torch.allclose(run_forward(model, cpu), logits))
    except Exception as error:
        record("CPU fallback", False, f"{type(error).__name__}: {error}")

    # 9 + 10. NaN / infinity
    outputs = [logits] + ([gpu_logits] if gpu_logits is not None else [])
    record("NaN check", not any(torch.isnan(t).any() for t in outputs))
    record("Infinity check", not any(torch.isinf(t).any() for t in outputs))

    # 11. Invalid input rejection
    invalid_ok = (
        rejects(lambda: model(torch.zeros(BATCH, 768)))            # Swin-sized vector
        and rejects(lambda: model(torch.zeros(INPUT_DIM)))         # no batch dimension
        and rejects(lambda: model(torch.zeros(BATCH, 49, INPUT_DIM)))  # tokens, not pooled
    )
    record("Invalid input rejection", invalid_ok, "[2,768], [256], [2,49,256] rejected")

    print()
    print(f"Input:  {list(INPUT_SHAPE)}")
    print(f"Output: {list(logits.shape)}  (index 0 = FAKE, index 1 = REAL)")
    print(f"Peak GPU memory: {peak_mb:.1f} MB" if peak_mb is not None else "Peak GPU memory: not measured")
    print("Weights: random - outputs are meaningless until training")

    all_passed = all(results.values())
    print("\nFinal result:")
    print("  PASS" if all_passed else "  FAIL")
    print("=" * 50)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
