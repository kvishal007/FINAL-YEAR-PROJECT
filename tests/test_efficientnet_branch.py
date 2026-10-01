"""
Read-only test for src/models/efficientnet_branch.py.

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
    """One forward pass on `device`; returns (pooled, feature_map, tokens) on CPU."""
    model = model.to(device)
    images = torch.randn(INPUT_SHAPE, device=device)
    with torch.inference_mode():
        pooled = model(images)
        feature_map = model.forward_feature_map(images)
        tokens = model.forward_tokens(images)
    if pooled.device.type != device.type:
        raise RuntimeError(f"Output on {pooled.device}, expected {device}")
    return pooled.cpu(), feature_map.cpu(), tokens.cpu()


def main():
    print("=" * 50)
    print("EFFICIENTNET BRANCH TEST")
    print("=" * 50)

    # 1. Import
    try:
        from src.models.efficientnet_branch import (
            EFFICIENTNET_FEATURE_DIM,
            EFFICIENTNET_NUM_TOKENS,
            EfficientNetBranch,
        )
        record("Import", True)
    except Exception as error:
        record("Import", False, f"{type(error).__name__}: {error}")
        return False

    # 2. Initialization (random weights -> no download)
    try:
        model = EfficientNetBranch(pretrained=False).eval()
        n_params = sum(p.numel() for p in model.parameters())
        record("Initialization", True, f"{n_params / 1e6:.1f}M parameters")
    except Exception as error:
        record("Initialization", False, f"{type(error).__name__}: {error}")
        return False

    # 3. Input shape: correct input accepted, wrong input rejected clearly
    try:
        model.forward_feature_map(torch.zeros(1, 1, 224, 224))  # 1 channel: invalid
        record("Input shape", False, "a 1-channel input was not rejected")
    except ValueError:
        record("Input shape", True, f"accepts {list(INPUT_SHAPE)}, rejects invalid shapes")

    # 4. Forward pass on CPU (always available)
    cpu = torch.device("cpu")
    try:
        cpu_pooled, cpu_map, cpu_tokens = run_forward(model, cpu)
        record("Forward pass", True)
    except Exception as error:
        record("Forward pass", False, f"{type(error).__name__}: {error}")
        return False

    batch = INPUT_SHAPE[0]
    grid = int(EFFICIENTNET_NUM_TOKENS ** 0.5)

    # 5. Feature representation: [B, 1280] that equals the pooled spatial map
    pooled_ok = (
        tuple(cpu_pooled.shape) == (batch, EFFICIENTNET_FEATURE_DIM)
        and torch.allclose(cpu_pooled, cpu_map.mean(dim=(2, 3)), atol=1e-5)
    )
    record("Feature representation", pooled_ok, f"shape {list(cpu_pooled.shape)}")

    # 6. Spatial feature representation: map + tokens hold the same values
    spatial_ok = (
        tuple(cpu_map.shape) == (batch, EFFICIENTNET_FEATURE_DIM, grid, grid)
        and tuple(cpu_tokens.shape) == (batch, EFFICIENTNET_NUM_TOKENS, EFFICIENTNET_FEATURE_DIM)
        and torch.equal(cpu_tokens, cpu_map.flatten(2).transpose(1, 2))
    )
    record(
        "Spatial feature representation",
        spatial_ok,
        f"map {list(cpu_map.shape)}, tokens {list(cpu_tokens.shape)}",
    )

    # 7. CUDA
    peak_mb = None
    gpu_outputs = []
    device_used = "CPU"
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        try:
            torch.cuda.reset_peak_memory_stats()
            gpu_outputs = list(run_forward(model, gpu))
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            same_shapes = all(
                g.shape == c.shape
                for g, c in zip(gpu_outputs, (cpu_pooled, cpu_map, cpu_tokens))
            )
            device_used = torch.cuda.get_device_name(0)
            record("CUDA", same_shapes, f"{device_used}, peak memory {peak_mb:.0f} MB")
        except Exception as error:
            record("CUDA", False, f"{type(error).__name__}: {error}")
        finally:
            model = model.to(cpu)
            torch.cuda.empty_cache()
    else:
        record("CUDA", False, "CUDA not available on this machine")

    # 8. CPU fallback: model moved back from GPU still works on CPU
    try:
        fallback_pooled, _, _ = run_forward(model, cpu)
        record("CPU fallback", fallback_pooled.shape == cpu_pooled.shape)
    except Exception as error:
        record("CPU fallback", False, f"{type(error).__name__}: {error}")

    # 9 + 10. NaN / infinity checks on every output produced
    outputs = [cpu_pooled, cpu_map, cpu_tokens, *gpu_outputs]
    record("NaN check", not any(torch.isnan(t).any() for t in outputs))
    record("Infinity check", not any(torch.isinf(t).any() for t in outputs))

    # 11. No ImageNet classification head anywhere in the module
    has_imagenet_head = any(
        isinstance(m, torch.nn.Linear) and m.out_features == IMAGENET_CLASSES
        for m in model.modules()
    )
    no_1000_output = all(t.shape[-1] != IMAGENET_CLASSES for t in outputs)
    record("No ImageNet classification head", not has_imagenet_head and no_1000_output)

    # Extra: freeze option stops gradients and keeps BatchNorm statistics fixed
    frozen = EfficientNetBranch(pretrained=False, freeze=True).train()
    bn_layers = [m for m in frozen.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    freeze_ok = (
        not any(p.requires_grad for p in frozen.parameters())
        and all(not bn.training for bn in bn_layers)
    )
    record("Freeze option", freeze_ok, "no trainable params, BatchNorm stays in eval")

    print()
    print(f"Input shape:\n  {list(INPUT_SHAPE)}")
    print(f"\nFeature shape:\n  forward()             -> {list(cpu_pooled.shape)}")
    print("\nSpatial feature shape:")
    print(f"  forward_feature_map() -> {list(cpu_map.shape)}")
    print(f"  forward_tokens()      -> {list(cpu_tokens.shape)}")
    print("\nGPU memory if measured:")
    print(f"  {peak_mb:.0f} MB peak (batch of {batch}, no gradients)" if peak_mb else "  not measured")
    print(f"\nDevice:\n  {device_used}")
    print("\nWeights: random (pretrained=False) - structure test only, not accuracy")

    all_passed = all(results.values())
    print("\nFinal result:")
    print("  PASS" if all_passed else "  FAIL")
    print("=" * 50)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
