"""
Read-only structural test for src/models/defax.py.

Random weights (pretrained=False) and a random dummy batch [2, 3, 224, 224]:
no downloads, no dataset, no training. Runs in eval mode inside
torch.no_grad(), so no gradients are stored.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

BATCH = 2
INPUT_SHAPE = (BATCH, 3, 224, 224)
NUM_HEADS = 8
IMAGENET_CLASSES = 1000

EXPECTED_FEATURES = {
    "swin_tokens": (BATCH, 49, 768),
    "efficientnet_tokens": (BATCH, 49, 1280),
    "fused_tokens": (BATCH, 49, 256),
    "fused": (BATCH, 256),
    "logits": (BATCH, 2),
}

results = {}


def record(name, passed, note=""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({note})" if note else ""))


def run_forward(model, device):
    """Full forward pass with all outputs; returns dict of CPU tensors."""
    model = model.to(device)
    torch.manual_seed(0)
    images = torch.randn(INPUT_SHAPE).to(device)
    with torch.no_grad():
        outputs = model(images, return_features=True, return_attention=True)
    if outputs["logits"].device.type != device.type:
        raise RuntimeError(f"Output on {outputs['logits'].device}, expected {device}")
    return {name: t.cpu() for name, t in outputs.items()}


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    print("=" * 50)
    print("COMPLETE DEFAX TEST")
    print("=" * 50)

    # 1. Import
    try:
        from src.models.defax import DeFaX
        record("Import", True)
    except Exception as error:
        record("Import", False, f"{type(error).__name__}: {error}")
        return False

    # 2. Initialization (random weights; `pretrained` must be given explicitly)
    try:
        model = DeFaX(pretrained=False).eval()
        try:
            DeFaX()
            explicit = False
        except TypeError:
            explicit = True
        record("Initialization", explicit, "random weights; pretrained must be set explicitly")
    except Exception as error:
        record("Initialization", False, f"{type(error).__name__}: {error}")
        return False

    cpu = torch.device("cpu")
    torch.manual_seed(0)
    images = torch.randn(INPUT_SHAPE)

    # 3. Complete forward pass (default mode returns logits only)
    try:
        with torch.no_grad():
            logits = model(images)
        record("Complete forward pass", torch.is_tensor(logits))
    except Exception as error:
        record("Complete forward pass", False, f"{type(error).__name__}: {error}")
        return False

    # 4. Correct input shape: [2,3,224,224] accepted, wrong channels rejected
    try:
        with torch.no_grad():
            model(torch.zeros(BATCH, 1, 224, 224))
        record("Correct input shape", False, "1-channel input was not rejected")
    except ValueError:
        record("Correct input shape", True, f"accepts {list(INPUT_SHAPE)}, rejects 1-channel input")

    # 5. Logits shape
    record("Logits shape [B,2]", tuple(logits.shape) == (BATCH, 2), str(list(logits.shape)))

    # 6. Intermediate feature shapes + pooling identical to forward_fused()
    with torch.no_grad():
        features = model(images, return_features=True)
        reference_fused = model.fusion.forward_fused(
            features["swin_tokens"], features["efficientnet_tokens"]
        )
    shapes_ok = all(tuple(features[k].shape) == v for k, v in EXPECTED_FEATURES.items())
    consistent = (
        torch.allclose(features["fused"], reference_fused, atol=1e-6)
        and torch.allclose(features["logits"], logits, atol=1e-6)
        and "attention" not in features
    )
    record("Intermediate feature shapes", shapes_ok and consistent,
           "all 5 shapes correct; matches forward_fused() and default logits")

    # 7. Attention output when requested
    with torch.no_grad():
        with_attention = model(images, return_features=True, return_attention=True)
    attention = with_attention.get("attention")
    attention_ok = (
        attention is not None
        and tuple(attention.shape) == (BATCH, NUM_HEADS, 49, 49)
        and torch.allclose(attention.sum(dim=-1), torch.ones(1), atol=1e-5)
    )
    record("Attention output when requested",
           attention_ok,
           f"{list(attention.shape)}, rows sum to 1" if attention is not None else "missing")

    # 8. CUDA
    cpu_outputs = run_forward(model, cpu)
    gpu_outputs = {}
    peak_mb = None
    device_used = "CPU"
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_outputs = run_forward(model, torch.device("cuda"))
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            device_used = torch.cuda.get_device_name(0)
            same = all(gpu_outputs[k].shape == cpu_outputs[k].shape for k in cpu_outputs)
            close = torch.allclose(gpu_outputs["logits"], cpu_outputs["logits"], atol=1e-2)
            record("CUDA", same and close,
                   f"{device_used}, peak memory {peak_mb:.0f} MB, logits match CPU")
        except Exception as error:
            record("CUDA", False, f"{type(error).__name__}: {error}")
        finally:
            model = model.to(cpu)
            torch.cuda.empty_cache()
    else:
        record("CUDA", False, "CUDA not available on this machine")

    # 9. CPU fallback
    try:
        fallback = run_forward(model, cpu)
        record("CPU fallback", torch.allclose(fallback["logits"], cpu_outputs["logits"], atol=1e-6))
    except Exception as error:
        record("CPU fallback", False, f"{type(error).__name__}: {error}")

    # 10 + 11. NaN / infinity across every output tensor
    all_tensors = [logits, *features.values(), *cpu_outputs.values(), *gpu_outputs.values()]
    record("NaN check", not any(torch.isnan(t).any() for t in all_tensors))
    record("Infinity check", not any(torch.isinf(t).any() for t in all_tensors))

    # 12. Parameter count: total = sum of the four parts, nothing extra
    parts = {
        "Swin-T": count(model.swin),
        "EfficientNet-B0": count(model.efficientnet),
        "Cross-attention": count(model.fusion),
        "Classifier": count(model.classifier),
    }
    total = count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record("Parameter count", total == sum(parts.values()) and trainable == total,
           f"{total / 1e6:.2f}M total, all trainable")

    # 13. No ImageNet classification head anywhere in the complete model
    imagenet_heads = [
        name for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and m.out_features == IMAGENET_CLASSES
    ]
    record("No accidental ImageNet classification heads", not imagenet_heads,
           "no Linear layer with 1000 outputs" if not imagenet_heads else str(imagenet_heads))

    print()
    print(f"Input:  {list(INPUT_SHAPE)}")
    print("\nOutputs (return_features=True, return_attention=True):")
    for name, tensor in with_attention.items():
        print(f"  {name:20} {list(tensor.shape)}")
    print("\nParameters:")
    for name, n in parts.items():
        print(f"  {name:16} {n:>12,}")
    print(f"  {'TOTAL':16} {total:>12,}")
    print(f"\nDevice: {device_used}")
    print(f"Peak GPU memory: {peak_mb:.0f} MB (batch {BATCH}, torch.no_grad)"
          if peak_mb is not None else "Peak GPU memory: not measured")
    print("Weights: random - structure test only, predictions are meaningless")

    all_passed = all(results.values())
    print("\nFinal result:")
    print("  PASS" if all_passed else "  FAIL")
    print("=" * 50)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
