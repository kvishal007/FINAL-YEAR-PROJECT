"""
Read-only check that the ImageNet pretrained weights are cached on D: and
load correctly into the existing model classes.

- Downloading is BLOCKED during this test: every load must come from the cache.
- No dataset, no training, no GPU needed (everything stays on the CPU).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.hub
from torchvision.models import EfficientNet_B0_Weights, Swin_T_Weights

EXPECTED_TORCH_HOME = Path(r"D:\FINAL YEAR PROJECT\torch_cache")
OLD_C_CACHE = Path.home() / ".cache" / "torch"

WEIGHTS = {
    "Swin-T": Swin_T_Weights.IMAGENET1K_V1,
    "EfficientNet-B0": EfficientNet_B0_Weights.IMAGENET1K_V1,
}

results = {}


def record(name, passed, note=""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({note})" if note else ""))


def checkpoint_path(weights):
    filename = weights.url.rsplit("/", 1)[-1]
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def block_downloads():
    """Make any download attempt fail loudly, so cached loading is proven."""
    def no_download(url, *args, **kwargs):
        raise RuntimeError(f"Download attempted: {url}")
    torch.hub.download_url_to_file = no_download


def compare_to_checkpoint(branch, checkpoint):
    """Return (matching tensors, total tensors in branch) - values must be identical."""
    state = branch.state_dict()
    matching = sum(
        1 for key, tensor in state.items()
        if key in checkpoint and torch.equal(tensor, checkpoint[key])
    )
    return matching, len(state)


def differs_from_random(pretrained_branch, random_branch):
    """Fraction of floating-point weight tensors that differ from a random init."""
    pre, rnd = pretrained_branch.state_dict(), random_branch.state_dict()
    keys = [k for k, t in pre.items() if t.is_floating_point()]
    differing = sum(1 for k in keys if not torch.equal(pre[k], rnd[k]))
    return differing / len(keys)


def main():
    print("=" * 60)
    print("PRETRAINED WEIGHTS TEST")
    print("=" * 60)

    # 1. Cache directory on D: is configured and used
    torch_home = os.environ.get("TORCH_HOME")
    hub_dir = Path(torch.hub.get_dir())
    cache_ok = (
        torch_home is not None
        and Path(torch_home) == EXPECTED_TORCH_HOME
        and EXPECTED_TORCH_HOME.is_dir()
        and EXPECTED_TORCH_HOME in hub_dir.parents
    )
    note = (f"torch.hub uses {hub_dir}" if cache_ok else
            f"TORCH_HOME={torch_home!r} - open a NEW terminal after setting it")
    record("D: cache directory exists", cache_ok, note)
    if not cache_ok:
        return False

    files = {name: checkpoint_path(w) for name, w in WEIGHTS.items()}
    if not all(p.is_file() for p in files.values()):
        missing = [str(p) for p in files.values() if not p.is_file()]
        record("Weight files present", False, f"missing: {missing}")
        return False
    stamps_before = {p: (p.stat().st_size, p.stat().st_mtime) for p in files.values()}

    block_downloads()

    from src.models.defax import DeFaX
    from src.models.efficientnet_branch import EfficientNetBranch
    from src.models.swin_branch import SwinBranch

    checkpoints = {
        name: torch.load(p, map_location="cpu", weights_only=True)
        for name, p in files.items()
    }

    # 2 + 3. Each branch loads its pretrained weights from the cache
    branches = {}
    for name, cls in (("Swin-T", SwinBranch), ("EfficientNet-B0", EfficientNetBranch)):
        label = "Swin pretrained weights load" if name == "Swin-T" else \
                "EfficientNet pretrained weights load"
        try:
            branches[name] = cls(pretrained=True).eval()
            matching, total = compare_to_checkpoint(branches[name], checkpoints[name])
            record(label, matching == total, f"{matching}/{total} tensors identical to the checkpoint")
        except Exception as error:
            record(label, False, f"{type(error).__name__}: {error}")

    # 4. No download on second load: loads succeeded with downloads blocked,
    #    and the cached files were not rewritten
    stamps_after = {p: (p.stat().st_size, p.stat().st_mtime) for p in files.values()}
    no_download = len(branches) == 2 and stamps_before == stamps_after
    record("No download required on second load", no_download,
           "loaded with downloads blocked; cache files unchanged")

    # 5. Complete model with pretrained backbones
    try:
        model = DeFaX(pretrained=True).eval()
        swin_match, swin_total = compare_to_checkpoint(model.swin, checkpoints["Swin-T"])
        eff_match, eff_total = compare_to_checkpoint(model.efficientnet, checkpoints["EfficientNet-B0"])
        init_ok = swin_match == swin_total and eff_match == eff_total
        record("Complete DeFaX(pretrained=True) initializes", init_ok,
               "both backbones inside DeFaX hold the pretrained weights")
    except Exception as error:
        record("Complete DeFaX(pretrained=True) initializes", False, f"{type(error).__name__}: {error}")
        model = None

    # 6 + 7. NaN / infinity across every parameter and buffer of the full model
    if model is not None:
        tensors = [t for t in model.state_dict().values() if t.is_floating_point()]
        record("No NaN parameters", not any(torch.isnan(t).any() for t in tensors),
               f"{len(tensors)} tensors checked")
        record("No infinite parameters", not any(torch.isinf(t).any() for t in tensors))
    else:
        record("No NaN parameters", False, "model did not initialize")
        record("No infinite parameters", False, "model did not initialize")

    # 8 + 9. Pretrained values really differ from a random initialization
    torch.manual_seed(0)
    random_swin = SwinBranch(pretrained=False)
    random_eff = EfficientNetBranch(pretrained=False)
    for name, random_branch, label in (
        ("Swin-T", random_swin, "Swin backbone contains pretrained non-random parameters"),
        ("EfficientNet-B0", random_eff, "EfficientNet backbone contains pretrained non-random parameters"),
    ):
        if name not in branches:
            record(label, False, "branch did not load")
            continue
        fraction = differs_from_random(branches[name], random_branch)
        matching, total = compare_to_checkpoint(branches[name], checkpoints[name])
        record(label, fraction > 0.95 and matching == total,
               f"{fraction:.0%} of weight tensors differ from random init")

    # Report
    total_mb = sum(p.stat().st_size for p in EXPECTED_TORCH_HOME.rglob("*") if p.is_file()) / 1024**2
    print(f"\nCache location:\n  {hub_dir / 'checkpoints'}")
    print("\nDownloaded files:")
    for name, p in files.items():
        print(f"  {p.name:45} {p.stat().st_size / 1024**2:6.1f} MB  ({name})")
    print(f"\nApproximate total cache size:\n  {total_mb:.1f} MB")
    print(f"\nOld C: cache ({OLD_C_CACHE}) exists:\n  {'YES' if OLD_C_CACHE.exists() else 'NO'}")
    print(f"\nSwin weights loaded:\n  {'YES' if results.get('Swin pretrained weights load') else 'NO'}")
    print(f"\nEfficientNet weights loaded:\n  "
          f"{'YES' if results.get('EfficientNet pretrained weights load') else 'NO'}")
    print(f"\nComplete DeFaX initialization:\n  "
          f"{'PASS' if results.get('Complete DeFaX(pretrained=True) initializes') else 'FAIL'}")
    print("\nNote: only the two backbones are pretrained. Cross-attention and the")
    print("classifier are still random and will only learn during training.")

    all_passed = all(results.values())
    print("\nFinal result:")
    print("  PASS" if all_passed else "  FAIL")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
