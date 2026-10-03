"""
LEVEL 4.3 - held-out TEST SET evaluation of a trained checkpoint.

Inference only: no optimizer, no backward pass, no parameter updates.
Only the test split is opened - the training and validation folders are
never listed or read.

Usage (from the project root):
    python -m src.evaluation.evaluate_test
    python -m src.evaluation.evaluate_test --checkpoint models/checkpoints/best.pt --name stage1

Outputs (ignored by Git):
    results/<name>_test_evaluation.txt     human-readable report
    results/<name>_test_metrics.json       the same numbers, machine-readable
    results/<name>_test_confusion_matrix.png

Labels: 0 = FAKE, 1 = REAL.
"""

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.hub
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms as T

from src.data.dataset import DATASET_ROOT
from src.data.transforms import valid_test_transform
from src.models.defax import DeFaX
from src.training.trainer import CLASS_NAMES, evaluate

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
BATCH_SIZE = 32
EXPECTED_TEST_SIZE = 20_000
EXPECTED_PER_CLASS = 10_000

# Chart colours (single-hue sequential blue ramp + neutral ink)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_downloads():
    def no_download(url, *args, **kwargs):
        raise RuntimeError(f"Download attempted (blocked): {url}")
    torch.hub.download_url_to_file = no_download


def fail(message):
    print(f"\nABORT: {message}")
    sys.exit(1)


def load_model(checkpoint_path, device):
    """Rebuild DeFaX with random weights, then load EVERY weight from the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names") != CLASS_NAMES:
        fail(f"checkpoint label map {checkpoint.get('class_names')} != {CLASS_NAMES}")

    config = checkpoint["config"]
    model = DeFaX(pretrained=False, freeze_backbones=config.get("freeze_backbones", False))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)  # inference only
    return model.to(device).eval(), checkpoint


def build_test_loader(use_cuda):
    random_ops = [t for t in valid_test_transform.transforms
                  if type(t).__name__.startswith("Random")]
    if random_ops:
        fail(f"test transform contains random operations: {random_ops}")

    test_set = datasets.ImageFolder(DATASET_ROOT / "test", transform=valid_test_transform)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=use_cuda)
    return test_set, loader


def weights_unchanged(model, checkpoint):
    """Every parameter and buffer still equals the checkpoint (proves no updates)."""
    saved = checkpoint["model_state_dict"]
    current = model.state_dict()
    return all(torch.equal(current[k].cpu(), saved[k]) for k in saved)


def build_report(info, stats):
    cm = stats["confusion"]
    m = stats["class_metrics"]
    support = {"FAKE": sum(cm[0]), "REAL": sum(cm[1])}
    correct = stats["correct"]
    total = stats["total"]

    lines = [
        "=" * 60,
        "LEVEL 4.3 — STAGE 1 TEST EVALUATION",
        "=" * 60,
        "Stage 1 held-out test performance (test set unused during training",
        "and model selection). Not a final project result.",
        "",
        f"Checkpoint:      {info['checkpoint']}",
        f"Epoch:           {info['epoch']}",
        f"Validation loss: {info['val_loss']:.4f}",
        "",
        f"Test samples:    {total:,}",
        f"FAKE:            {info['test_fake']:,}",
        f"REAL:            {info['test_real']:,}",
        "",
        f"Test loss:       {stats['loss']:.4f}",
        f"Test accuracy:   {stats['accuracy']:.2%}",
        f"Correct:         {correct:,}",
        f"Incorrect:       {total - correct:,}",
        "",
    ]
    for name in ("FAKE", "REAL"):
        lines += [
            name,
            f"Precision: {m[name]['precision']:.4f}",
            f"Recall:    {m[name]['recall']:.4f}",
            f"F1:        {m[name]['f1']:.4f}",
            f"Support:   {support[name]:,}",
            "",
        ]
    lines += [
        "Confusion Matrix:",
        "",
        f"{'':14}{'Pred FAKE':>12}{'Pred REAL':>12}",
        f"{'Actual FAKE':14}{cm[0][0]:>12,}{cm[0][1]:>12,}",
        f"{'Actual REAL':14}{cm[1][0]:>12,}{cm[1][1]:>12,}",
        "",
        f"FAKE images wrongly accepted as REAL (missed fakes): {cm[0][1]:,}",
        f"REAL images wrongly flagged as FAKE (false alarms):  {cm[1][0]:,}",
        "",
        "-" * 60,
        f"Device:            {info['device']}",
        f"PyTorch:           {torch.__version__}",
        f"Batch size:        {BATCH_SIZE} (no shuffling, no augmentation, AMP {info['amp']})",
        f"Checkpoint SHA-256: {info['sha_before']}",
        f"Checkpoint unchanged after evaluation: {info['sha_before'] == info['sha_after']}",
        f"Model weights unchanged by evaluation: {info['weights_unchanged']}",
        f"Evaluated at:      {info['timestamp']}",
        "",
        "=" * 60,
        "TEST EVALUATION COMPLETE",
        "=" * 60,
    ]
    return "\n".join(lines)


def save_confusion_matrix(cm, accuracy, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    names = [CLASS_NAMES[0], CLASS_NAMES[1]]
    row_totals = [sum(row) for row in cm]

    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(row_totals))

    # 2px surface gap between cells
    for edge in (0.5,):
        ax.axhline(edge, color=SURFACE, linewidth=3)
        ax.axvline(edge, color=SURFACE, linewidth=3)

    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            dark_cell = value / max(row_totals) > 0.5
            colour = "#ffffff" if dark_cell else INK_PRIMARY
            ax.text(j, i - 0.08, f"{value:,}", ha="center", va="center",
                    fontsize=17, fontweight="bold", color=colour)
            ax.text(j, i + 0.16, f"{value / row_totals[i]:.2%} of actual {names[i]}",
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark_cell else INK_SECONDARY)

    ax.set_xticks([0, 1], [f"Predicted {n}" for n in names], color=INK_SECONDARY, fontsize=10)
    ax.set_yticks([0, 1], [f"Actual {n}" for n in names], color=INK_SECONDARY, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=22)
    ax.text(0, 1.03, f"{sum(row_totals):,} held-out test images · accuracy {accuracy:.2%} · "
            f"darker = more images", transform=ax.transAxes, color=INK_MUTED, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DeFaX checkpoint on the held-out test set")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--name", default="stage1", help="prefix for the result files")
    args = parser.parse_args()

    block_downloads()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = use_cuda

    checkpoint_path = Path(args.checkpoint).resolve()
    print("=" * 60)
    print("PRE-EVALUATION CHECKS")
    print("=" * 60)

    # 1. Checkpoint
    if not checkpoint_path.is_file():
        fail(f"checkpoint not found: {checkpoint_path}")
    sha_before = sha256(checkpoint_path)
    print(f"[PASS] Checkpoint exists: {checkpoint_path}")

    try:
        model, checkpoint = load_model(checkpoint_path, device)
    except Exception as error:
        fail(f"model did not load: {type(error).__name__}: {error}")
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["metrics"]["val_loss"]
    print(f"[PASS] Checkpoint epoch: {epoch}")
    print(f"[PASS] Checkpoint validation loss: {val_loss:.4f}")
    print(f"[PASS] Model loaded (strict, all {len(checkpoint['model_state_dict'])} tensors), "
          f"label map {checkpoint['class_names']}, on {device}")

    # 2. Test data
    test_set, test_loader = build_test_loader(use_cuda)
    counts = {CLASS_NAMES[c]: test_set.targets.count(c) for c in CLASS_NAMES}
    if test_set.class_to_idx != {"fake": 0, "real": 1}:
        fail(f"unexpected class_to_idx {test_set.class_to_idx}")
    if len(test_set) != EXPECTED_TEST_SIZE:
        fail(f"test set has {len(test_set):,} images, expected {EXPECTED_TEST_SIZE:,}")
    if counts != {"FAKE": EXPECTED_PER_CLASS, "REAL": EXPECTED_PER_CLASS}:
        fail(f"class distribution {counts}")
    if not isinstance(test_loader.sampler, SequentialSampler):
        fail("test loader shuffles")
    print(f"[PASS] Test set: {len(test_set):,} images, {counts}, class_to_idx {test_set.class_to_idx}")
    print(f"[PASS] Test loader: batch {BATCH_SIZE}, no shuffling, transform has no random operations")

    # 3. Inference only
    print(f"\nEvaluating {len(test_loader)} batches (inference only)...", flush=True)
    criterion = nn.CrossEntropyLoss()
    stats = evaluate(model, test_loader, criterion, device, use_amp)

    unchanged = weights_unchanged(model, checkpoint)
    sha_after = sha256(checkpoint_path)
    if stats["total"] != EXPECTED_TEST_SIZE:
        fail(f"evaluated {stats['total']} images, expected {EXPECTED_TEST_SIZE}")

    info = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "epoch": epoch,
        "val_loss": val_loss,
        "test_fake": counts["FAKE"],
        "test_real": counts["REAL"],
        "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "amp": use_amp,
        "sha_before": sha_before,
        "sha_after": sha_after,
        "weights_unchanged": unchanged,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    report = build_report(info, stats)
    print("\n" + report)

    # 4. Save results (results/ only)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"{args.name}_test_evaluation.txt"
    metrics_path = RESULTS_DIR / f"{args.name}_test_metrics.json"
    figure_path = RESULTS_DIR / f"{args.name}_test_confusion_matrix.png"

    report_path.write_text(report + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps({
        "description": f"{args.name} held-out test performance",
        "checkpoint": info["checkpoint"],
        "checkpoint_sha256": sha_before,
        "epoch": epoch,
        "validation_loss": val_loss,
        "test_samples": stats["total"],
        "test_loss": stats["loss"],
        "test_accuracy": stats["accuracy"],
        "correct": stats["correct"],
        "incorrect": stats["total"] - stats["correct"],
        "class_metrics": stats["class_metrics"],
        "support": {"FAKE": sum(stats["confusion"][0]), "REAL": sum(stats["confusion"][1])},
        "confusion_matrix": {"rows": "actual [FAKE, REAL]", "cols": "predicted [FAKE, REAL]",
                             "values": stats["confusion"]},
        "label_map": {str(k): v for k, v in CLASS_NAMES.items()},
        "evaluated_at": info["timestamp"],
    }, indent=2), encoding="utf-8")
    readable_name = re.sub(r"(\D)(\d)", r"\1 \2", args.name).capitalize()  # "stage1" -> "Stage 1"
    save_confusion_matrix(stats["confusion"], stats["accuracy"], figure_path,
                          f"{readable_name} held-out test set: confusion matrix")

    print("\nSaved:")
    for path in (report_path, metrics_path, figure_path):
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    if not (unchanged and sha_before == sha_after):
        print("WARNING: checkpoint or weights changed during evaluation")
        return False
    return True


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    sys.exit(0 if main() else 1)
