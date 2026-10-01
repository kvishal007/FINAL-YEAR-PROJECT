"""
Read-only test for src/models/cross_attention.py.

Uses random dummy tokens shaped like the real branch outputs, so no images,
weights or dataset are needed. No training happens: the module runs in eval
mode inside torch.inference_mode().
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

BATCH = 2
NUM_TOKENS = 49
SWIN_DIM = 768
EFFICIENTNET_DIM = 1280
ATTENTION_DIM = 256
NUM_HEADS = 8

SWIN_SHAPE = (BATCH, NUM_TOKENS, SWIN_DIM)
EFFICIENTNET_SHAPE = (BATCH, NUM_TOKENS, EFFICIENTNET_DIM)

results = {}


def record(name, passed, note=""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({note})" if note else ""))


def rejects(fn):
    """True if fn() raises ValueError."""
    try:
        fn()
        return False
    except ValueError:
        return True


def run_forward(model, device):
    """Run the module on `device`; returns (tokens, fused) on CPU."""
    model = model.to(device)
    torch.manual_seed(0)
    swin = torch.randn(SWIN_SHAPE).to(device)
    eff = torch.randn(EFFICIENTNET_SHAPE).to(device)
    with torch.inference_mode():
        tokens = model(swin, eff)
        fused = model.forward_fused(swin, eff)
    if tokens.device.type != device.type:
        raise RuntimeError(f"Output on {tokens.device}, expected {device}")
    return tokens.cpu(), fused.cpu()


def main():
    print("=" * 50)
    print("CROSS-ATTENTION TEST")
    print("=" * 50)

    # 1. Import
    try:
        from src.models.cross_attention import CrossAttentionFusion
        record("Import", True)
    except Exception as error:
        record("Import", False, f"{type(error).__name__}: {error}")
        return False

    # 2. Initialization
    try:
        model = CrossAttentionFusion(
            swin_dim=SWIN_DIM,
            efficientnet_dim=EFFICIENTNET_DIM,
            attention_dim=ATTENTION_DIM,
            num_heads=NUM_HEADS,
        ).eval()
        n_params = sum(p.numel() for p in model.parameters())
        record("Initialization", True, f"{n_params / 1e6:.2f}M parameters")
    except Exception as error:
        record("Initialization", False, f"{type(error).__name__}: {error}")
        return False

    torch.manual_seed(0)
    swin = torch.randn(SWIN_SHAPE)
    eff = torch.randn(EFFICIENTNET_SHAPE)

    # 3 + 4. Input validation for each stream
    good_eff = torch.zeros(EFFICIENTNET_SHAPE)
    good_swin = torch.zeros(SWIN_SHAPE)
    swin_ok = (
        rejects(lambda: model(torch.zeros(BATCH, NUM_TOKENS, 512), good_eff))   # wrong dim
        and rejects(lambda: model(torch.zeros(NUM_TOKENS, SWIN_DIM), good_eff))  # no batch
    )
    record("Swin input shape", swin_ok, f"accepts {list(SWIN_SHAPE)}, rejects wrong shapes")

    eff_ok = (
        rejects(lambda: model(good_swin, torch.zeros(BATCH, NUM_TOKENS, 768)))   # Swin-sized
        and rejects(lambda: model(good_swin, torch.zeros(3, NUM_TOKENS, EFFICIENTNET_DIM)))  # batch mismatch
    )
    record("EfficientNet input shape", eff_ok, f"accepts {list(EFFICIENTNET_SHAPE)}, rejects wrong shapes")

    # 5. Dimension projections: both streams mapped into the same space
    proj_ok = (
        (model.q_proj.in_features, model.q_proj.out_features) == (SWIN_DIM, ATTENTION_DIM)
        and (model.k_proj.in_features, model.k_proj.out_features) == (EFFICIENTNET_DIM, ATTENTION_DIM)
        and (model.v_proj.in_features, model.v_proj.out_features) == (EFFICIENTNET_DIM, ATTENTION_DIM)
    )
    record("Dimension projections", proj_ok, f"768 -> {ATTENTION_DIM}, 1280 -> {ATTENTION_DIM}")

    # 6. Q/K/V generation: Q depends only on Swin, K/V only on EfficientNet
    with torch.inference_mode():
        q, k, v = model.project_qkv(swin, eff)
        q2, k2, v2 = model.project_qkv(swin, torch.randn(EFFICIENTNET_SHAPE))  # new EfficientNet
        q3, k3, v3 = model.project_qkv(torch.randn(SWIN_SHAPE), eff)           # new Swin
    expected = (BATCH, NUM_TOKENS, ATTENTION_DIM)
    qkv_ok = (
        all(tuple(t.shape) == expected for t in (q, k, v))
        and torch.equal(q, q2) and not torch.equal(k, k2) and not torch.equal(v, v2)
        and torch.equal(k, k3) and torch.equal(v, v3) and not torch.equal(q, q3)
    )
    record("Q/K/V generation", qkv_ok, "Q from Swin, K/V from EfficientNet, each " + str(list(expected)))

    # 7. Multi-head cross-attention: weight shape, rows sum to 1, matches the
    #    PyTorch reference attention, and the output really uses EfficientNet
    with torch.inference_mode():
        tokens, attention = model(swin, eff, return_attention=True)

        split = lambda t: t.view(BATCH, -1, NUM_HEADS, ATTENTION_DIM // NUM_HEADS).transpose(1, 2)
        reference = F.scaled_dot_product_attention(split(q), split(k), split(v))
        reference = reference.transpose(1, 2).reshape(BATCH, NUM_TOKENS, ATTENTION_DIM)
        reference = model.norm(q + model.out_proj(reference))

        tokens_new_eff = model(swin, torch.randn(EFFICIENTNET_SHAPE))

    attn_ok = (
        tuple(attention.shape) == (BATCH, NUM_HEADS, NUM_TOKENS, NUM_TOKENS)
        and torch.allclose(attention.sum(dim=-1), torch.ones(1), atol=1e-5)
        and torch.allclose(tokens, reference, atol=1e-5)
        and not torch.allclose(tokens, tokens_new_eff)
    )
    record(
        "Multi-head cross-attention",
        attn_ok,
        f"{NUM_HEADS} heads x {ATTENTION_DIM // NUM_HEADS} dims, weights {list(attention.shape)}, "
        "matches PyTorch reference",
    )

    # 8. Output token shape
    record(
        "Output token shape",
        tuple(tokens.shape) == (BATCH, NUM_TOKENS, ATTENTION_DIM),
        str(list(tokens.shape)),
    )

    # 9. Fused representation: [B, D] equal to the mean of the tokens
    with torch.inference_mode():
        fused = model.forward_fused(swin, eff)
    fusion_ok = (
        tuple(fused.shape) == (BATCH, ATTENTION_DIM)
        and torch.allclose(fused, tokens.mean(dim=1), atol=1e-6)
    )
    record("Fused representation shape", fusion_ok, f"{list(fused.shape)}, mean over {NUM_TOKENS} tokens")

    # 10. CUDA
    cpu = torch.device("cpu")
    cpu_tokens, cpu_fused = run_forward(model, cpu)
    gpu_outputs = []
    peak_mb = None
    device_used = "CPU"
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            gpu_tokens, gpu_fused = run_forward(model, torch.device("cuda"))
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            gpu_outputs = [gpu_tokens, gpu_fused]
            device_used = torch.cuda.get_device_name(0)
            cuda_ok = torch.allclose(gpu_tokens, cpu_tokens, atol=1e-3)
            record("CUDA", cuda_ok, f"{device_used}, peak memory {peak_mb:.0f} MB, matches CPU result")
        except Exception as error:
            record("CUDA", False, f"{type(error).__name__}: {error}")
        finally:
            model = model.to(cpu)
            torch.cuda.empty_cache()
    else:
        record("CUDA", False, "CUDA not available on this machine")

    # 11. CPU fallback: module moved back from GPU still works on CPU
    try:
        fallback_tokens, _ = run_forward(model, cpu)
        record("CPU fallback", torch.allclose(fallback_tokens, cpu_tokens, atol=1e-6))
    except Exception as error:
        record("CPU fallback", False, f"{type(error).__name__}: {error}")

    # 12 + 13. NaN / infinity checks on every output produced
    outputs = [tokens, attention, fused, cpu_tokens, cpu_fused, *gpu_outputs]
    record("NaN check", not any(torch.isnan(t).any() for t in outputs))
    record("Infinity check", not any(torch.isinf(t).any() for t in outputs))

    # 14. Invalid attention dimension rejection
    invalid_ok = rejects(lambda: CrossAttentionFusion(attention_dim=250, num_heads=8))
    record("Invalid attention dimension rejection", invalid_ok, "attention_dim=250, num_heads=8 rejected")

    print()
    print("Input:")
    print(f"  Swin         = {list(SWIN_SHAPE)}")
    print(f"  EfficientNet = {list(EFFICIENTNET_SHAPE)}")
    print("\nOutput:")
    print(f"  Tokens       = {list(tokens.shape)}")
    print(f"  Fused        = {list(fused.shape)}")
    print(f"  Attention    = {list(attention.shape)}")
    print(f"\nConfiguration: attention_dim={ATTENTION_DIM}, num_heads={NUM_HEADS}, "
          f"head_dim={ATTENTION_DIM // NUM_HEADS}")
    print(f"Device: {device_used}" + (f", peak GPU memory {peak_mb:.0f} MB" if peak_mb else ""))
    print("Weights: random - structure test only, not accuracy")

    all_passed = all(results.values())
    print("\nFinal result:")
    print("  PASS" if all_passed else "  FAIL")
    print("=" * 50)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
