"""
Read-only test for src/models/swin_branch.py.

Uses random weights (pretrained=False) so nothing is downloaded, and a dummy
input, so the dataset is never touched. No training happens: the model runs
in eval mode inside torch.inference_mode(), so no gradients are stored.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

INPUT_SHAPE = (2, 3, 224, 224)
IMAGENET_CLASSES = 1000

results = {}


def record(name, passed, note=""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({note})" if note else ""))


def run_forward(model, device):
    """Run one forward pass on `device`; returns (pooled, tokens) on CPU."""
    model = model.to(device)
    images = torch.randn(INPUT_SHAPE, device=device)
    with torch.inference_mode():
        pooled = model(images)
        tokens = model.forward_tokens(images)
    if pooled.device.type != device.type:
        raise RuntimeError(f"Output on {pooled.device}, expected {device}")
    return pooled.cpu(), tokens.cpu()


def main():
    print("=" * 50)
    print("SWIN BRANCH TEST")
    print("=" * 50)

    # 1. Import
    try:
        from src.models.swin_branch import SWIN_FEATURE_DIM, SWIN_NUM_TOKENS, SwinBranch
        record("Import", True)
    except Exception as error:
        record("Import", False, f"{type(error).__name__}: {error}")
        return False

    # 2. Model initialization (random weights -> no download)
    try:
        model = SwinBranch(pretrained=False).eval()
        n_params = sum(p.numel() for p in model.parameters())
        record("Model initialization", True, f"{n_params / 1e6:.1f}M parameters")
    except Exception as error:
        record("Model initialization", False, f"{type(error).__name__}: {error}")
        return False

    # 3. Input shape: correct input accepted, wrong input rejected clearly
    try:
        model.forward_tokens(torch.zeros(1, 1, 224, 224))  # 1 channel: invalid
        record("Input shape", False, "a 1-channel input was not rejected")
    except ValueError:
        record("Input shape", True, f"accepts {list(INPUT_SHAPE)}, rejects invalid shapes")

    # 4. Forward pass on CPU (always available)
    cpu = torch.device("cpu")
    try:
        cpu_pooled, cpu_tokens = run_forward(model, cpu)
        record("Forward pass", True)
    except Exception as error:
        record("Forward pass", False, f"{type(error).__name__}: {error}")
        return False

    # 5. Feature output: 768-dim features, not 1000 ImageNet class scores
    has_imagenet_head = any(
        isinstance(m, torch.nn.Linear) and m.out_features == IMAGENET_CLASSES
        for m in model.modules()
    )
    expected_pooled = (INPUT_SHAPE[0], SWIN_FEATURE_DIM)
    expected_tokens = (INPUT_SHAPE[0], SWIN_NUM_TOKENS, SWIN_FEATURE_DIM)
    feature_ok = (
        tuple(cpu_pooled.shape) == expected_pooled
        and tuple(cpu_tokens.shape) == expected_tokens
        and not has_imagenet_head
    )
    record(
        "Feature output",
        feature_ok,
        "no 1000-class head; shapes match documentation"
        if feature_ok else
        f"got {list(cpu_pooled.shape)} / {list(cpu_tokens.shape)}, "
        f"ImageNet head present: {has_imagenet_head}",
    )

    # 6. CUDA
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        try:
            torch.cuda.reset_peak_memory_stats()
            gpu_pooled, gpu_tokens = run_forward(model, gpu)
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            same_shape = (gpu_pooled.shape == cpu_pooled.shape
                          and gpu_tokens.shape == cpu_tokens.shape)
            record(
                "CUDA",
                same_shape,
                f"{torch.cuda.get_device_name(0)}, peak memory {peak_mb:.0f} MB",
            )
            device_used = torch.cuda.get_device_name(0)
        except Exception as error:
            record("CUDA", False, f"{type(error).__name__}: {error}")
            gpu_pooled = gpu_tokens = None
            device_used = "CPU"
        finally:
            model = model.to(cpu)
            torch.cuda.empty_cache()
    else:
        record("CUDA", False, "CUDA not available on this machine")
        gpu_pooled = gpu_tokens = None
        device_used = "CPU"

    # 7. CPU fallback: model moved back from GPU still works on CPU
    try:
        fallback_pooled, _ = run_forward(model, cpu)
        record("CPU fallback", fallback_pooled.shape == cpu_pooled.shape)
    except Exception as error:
        record("CPU fallback", False, f"{type(error).__name__}: {error}")

    # 8 + 9. NaN / infinity checks on every output produced
    outputs = [t for t in (cpu_pooled, cpu_tokens, gpu_pooled, gpu_tokens) if t is not None]
    record("NaN check", not any(torch.isnan(t).any() for t in outputs))
    record("Infinity check", not any(torch.isinf(t).any() for t in outputs))

    print()
    print(f"Input shape:\n  {list(INPUT_SHAPE)}")
    print(f"\nOutput shape:")
    print(f"  forward()        -> {list(cpu_pooled.shape)}   (global feature vector)")
    print(f"  forward_tokens() -> {list(cpu_tokens.shape)}   (7x7 spatial tokens)")
    print(f"\nDevice:\n  {device_used}")
    print("\nWeights: random (pretrained=False) - structure test only, not accuracy")

    all_passed = all(results.values())
    print("\nFinal result:")
    print("  PASS" if all_passed else "  FAIL")
    print("=" * 50)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
