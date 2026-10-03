"""
LEVEL 4.4 - deterministic error analysis on the held-out TEST set.

Analysis only: no training, no optimizer, no backward pass, no weight
changes. Inference matches Level 4.3 exactly (same checkpoint loading, test
loader, transform and AMP setting), and the error counts are cross-checked
against results/<name>_test_metrics.json from that evaluation.

For every test image it records the prediction, class probabilities and three
cheap image properties computed from the already-loaded tensor:
    brightness   mean grey level of the 224x224 model input (0-1)
    sharpness    variance of the Laplacian of the grey image (higher = more fine detail)
    file_kb      size of the JPEG file on disk

Usage (from the project root):
    python -m src.evaluation.error_analysis

Outputs (deterministic - no timestamps - so two runs give identical files):
    results/error_analysis/all_predictions.csv     every test image
    results/error_analysis/errors.csv              all misclassified images, most confident first
    results/error_analysis/missed_fakes.csv        true FAKE -> predicted REAL
    results/error_analysis/false_alarms.csv        true REAL -> predicted FAKE
    results/error_analysis/summary.json / summary.txt
    results/error_analysis/sheet_<group>_<selection>.png   contact sheets

Labels: 0 = FAKE, 1 = REAL.
"""

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.evaluation.evaluate_test import (
    DEFAULT_CHECKPOINT,
    EXPECTED_PER_CLASS,
    EXPECTED_TEST_SIZE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    RESULTS_DIR,
    SURFACE,
    block_downloads,
    build_test_loader,
    fail,
    load_model,
    sha256,
    weights_unchanged,
)
from src.training.trainer import CLASS_NAMES

FAKE, REAL = 0, 1
ERROR_GROUPS = {
    "missed_fake": {"true": FAKE, "pred": REAL,
                    "title": "Missed fakes", "meaning": "true FAKE predicted REAL"},
    "false_alarm": {"true": REAL, "pred": FAKE,
                    "title": "False alarms", "meaning": "true REAL predicted FAKE"},
}
CONFIDENCE_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0000001]
SHEET_SIZE = 24          # images per contact sheet (6 x 4)
SHEET_COLUMNS = 6
THUMB = 160              # thumbnail edge in pixels
FLOAT_DIGITS = 6         # rounding in CSV/JSON keeps files byte-identical across runs

# ImageNet normalisation used by valid_test_transform (needed to undo it)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


# ------------------------------------------------------------------ inference

def image_properties(images):
    """Brightness and sharpness per image from normalised model inputs [B,3,H,W]."""
    rgb = (images.float() * STD.to(images.device) + MEAN.to(images.device)).clamp(0, 1)
    grey = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    brightness = grey.mean(dim=(1, 2, 3))
    edges = F.conv2d(grey * 255.0, LAPLACIAN.to(images.device))
    sharpness = edges.var(dim=(1, 2, 3))
    return brightness, sharpness


@torch.no_grad()
def predict_all(model, loader, dataset, device, use_amp):
    """One record per test image, in dataset order (the loader does not shuffle)."""
    records = []
    index = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probs = logits.float().softmax(dim=1).cpu()
        predictions = logits.argmax(dim=1).cpu()
        brightness, sharpness = (t.cpu() for t in image_properties(images))

        for i in range(labels.size(0)):
            path, target = dataset.samples[index]
            if target != labels[i].item():
                fail(f"loader order does not match dataset.samples at index {index}")
            true_label, pred_label = int(target), int(predictions[i])
            records.append({
                "index": index,
                "image_path": Path(path).as_posix(),
                "true_label": true_label,
                "predicted_label": pred_label,
                "true_class": CLASS_NAMES[true_label],
                "predicted_class": CLASS_NAMES[pred_label],
                "correct": true_label == pred_label,
                "confidence": float(probs[i, pred_label]),
                "fake_probability": float(probs[i, FAKE]),
                "real_probability": float(probs[i, REAL]),
                "brightness": float(brightness[i]),
                "sharpness": float(sharpness[i]),
                "file_kb": os.path.getsize(path) / 1024,
            })
            index += 1
    return records


def error_type(record):
    for name, group in ERROR_GROUPS.items():
        if record["true_label"] == group["true"] and record["predicted_label"] == group["pred"]:
            return name
    return ""


# ------------------------------------------------------------------ statistics

def describe(values):
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def confidence_bins(records):
    """Count predictions and accuracy per confidence band (a simple calibration view)."""
    bins = []
    for low, high in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        inside = [r for r in records if low <= r["confidence"] < high]
        correct = sum(r["correct"] for r in inside)
        bins.append({
            "range": f"{low:.2f}-{min(high, 1.0):.2f}",
            "predictions": len(inside),
            "errors": len(inside) - correct,
            "accuracy": correct / len(inside) if inside else None,
        })
    return bins


def rounded(value):
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def build_summary(records, checkpoint_info):
    errors = [r for r in records if not r["correct"]]
    correct = [r for r in records if r["correct"]]
    per_true = {c: sum(r["true_label"] == c for r in records) for c in CLASS_NAMES}

    summary = {
        "description": "Stage 1 held-out test error analysis (analysis only, no training)",
        "checkpoint": checkpoint_info,
        "total_test_images": len(records),
        "total_correct": len(correct),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(records),
        "groups": {},
        "confidence_bins_all_predictions": confidence_bins(records),
        "image_properties": {},
    }
    for name, group in ERROR_GROUPS.items():
        members = [r for r in errors if r["error_type"] == name]
        summary["groups"][name] = {
            "meaning": group["meaning"],
            "count": len(members),
            "rate_within_true_class": len(members) / per_true[group["true"]],
            "share_of_all_errors": len(members) / len(errors) if errors else 0.0,
            "confidence": describe([r["confidence"] for r in members]),
            "high_confidence_count_ge_0.99": sum(r["confidence"] >= 0.99 for r in members),
            "high_confidence_count_ge_0.90": sum(r["confidence"] >= 0.90 for r in members),
            "low_confidence_count_lt_0.60": sum(r["confidence"] < 0.60 for r in members),
        }
    summary["correct_predictions_confidence"] = describe([r["confidence"] for r in correct])

    # Do errors look different from correctly classified images of the SAME true class?
    for prop in ("brightness", "sharpness", "file_kb"):
        summary["image_properties"][prop] = {}
        for name, group in ERROR_GROUPS.items():
            same_class_correct = [r[prop] for r in correct if r["true_label"] == group["true"]]
            members = [r[prop] for r in errors if r["error_type"] == name]
            summary["image_properties"][prop][name] = describe(members)
            summary["image_properties"][prop][f"correct_{CLASS_NAMES[group['true']]}"] = \
                describe(same_class_correct)
    return rounded(summary)


# ------------------------------------------------------------------ files

CSV_COLUMNS = ["rank", "image_path", "true_label", "predicted_label", "true_class",
               "predicted_class", "error_type", "confidence", "fake_probability",
               "real_probability", "brightness", "sharpness", "file_kb", "index"]


def by_confidence(records):
    """Most confident first; image path breaks ties so ordering is fully deterministic."""
    return sorted(records, key=lambda r: (-r["confidence"], r["image_path"]))


def write_csv(path, records, ranked=True):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for rank, r in enumerate(records, start=1):
            row = {k: rounded(r.get(k, "")) for k in CSV_COLUMNS if k != "rank"}
            row["rank"] = rank if ranked else ""
            writer.writerow(row)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def contact_sheet(records, title, subtitle, path):
    """Grid of thumbnails, each captioned with rank, P(fake) and file name."""
    rows = (len(records) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    pad, caption_h, header_h = 12, 34, 64
    width = SHEET_COLUMNS * (THUMB + pad) + pad
    height = header_h + rows * (THUMB + caption_h + pad) + pad
    sheet = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 12), title, fill=INK_PRIMARY, font=font(20))
    draw.text((pad, 40), subtitle, fill=INK_MUTED, font=font(13))

    for i, r in enumerate(records):
        x = pad + (i % SHEET_COLUMNS) * (THUMB + pad)
        y = header_h + (i // SHEET_COLUMNS) * (THUMB + caption_h + pad)
        with Image.open(PROJECT_ROOT / r["image_path"]) as img:
            sheet.paste(img.convert("RGB").resize((THUMB, THUMB), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + THUMB + 3),
                  f"#{r['rank']}  conf {r['confidence']:.3f}  P(fake) {r['fake_probability']:.3f}",
                  fill=INK_PRIMARY, font=font(11))
        draw.text((x, y + THUMB + 18), Path(r["image_path"]).name, fill=INK_SECONDARY, font=font(11))

    # Fixed PNG settings + no metadata -> byte-identical output on every run
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=False, compress_level=6)
    path.write_bytes(buffer.getvalue())


def representative(records, count):
    """Evenly spaced picks across the confidence-sorted list (deterministic)."""
    if len(records) <= count:
        return records
    positions = np.linspace(0, len(records) - 1, count).round().astype(int)
    return [records[p] for p in positions]


def summary_text(s):
    lines = [
        "=" * 64,
        "LEVEL 4.4 - STAGE 1 TEST ERROR ANALYSIS",
        "=" * 64,
        "Analysis of held-out test predictions. No training was performed.",
        f"Checkpoint: {s['checkpoint']['path']} (epoch {s['checkpoint']['epoch']})",
        f"SHA-256:    {s['checkpoint']['sha256']}",
        "",
        f"Total test images : {s['total_test_images']:,}",
        f"Correct           : {s['total_correct']:,}",
        f"Total errors      : {s['total_errors']:,}",
        f"Error rate        : {s['error_rate']:.2%}",
        "",
    ]
    for name, g in s["groups"].items():
        c = g["confidence"]
        lines += [
            f"{ERROR_GROUPS[name]['title']} ({g['meaning']})",
            f"  count                 : {g['count']:,}  ({g['rate_within_true_class']:.2%} of that class, "
            f"{g['share_of_all_errors']:.1%} of errors)",
            f"  confidence mean/median: {c['mean']:.4f} / {c['median']:.4f}  "
            f"(min {c['min']:.4f}, max {c['max']:.4f}, std {c['std']:.4f})",
            f"  confidence >= 0.99    : {g['high_confidence_count_ge_0.99']}",
            f"  confidence >= 0.90    : {g['high_confidence_count_ge_0.90']}",
            f"  confidence <  0.60    : {g['low_confidence_count_lt_0.60']}",
            "",
        ]
    cc = s["correct_predictions_confidence"]
    lines += [f"Correct predictions confidence mean/median: {cc['mean']:.4f} / {cc['median']:.4f}", "",
              "Accuracy by confidence band (all 20,000 predictions):",
              f"  {'band':<12}{'predictions':>12}{'errors':>9}{'accuracy':>11}"]
    for b in s["confidence_bins_all_predictions"]:
        acc = f"{b['accuracy']:.2%}" if b["accuracy"] is not None else "-"
        lines.append(f"  {b['range']:<12}{b['predictions']:>12,}{b['errors']:>9,}{acc:>11}")
    lines += ["", "Image properties (median): errors vs correctly classified images of the same true class"]
    for prop, groups in s["image_properties"].items():
        parts = [f"{k} {v['median']:.4g}" for k, v in groups.items() if v.get("count")]
        lines.append(f"  {prop:<11}: " + ", ".join(parts))
    lines += ["", "=" * 64]
    return "\n".join(lines)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Deterministic error analysis on the held-out test set")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--name", default="stage1", help="prefix of the Level 4.3 metrics file to cross-check")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "error_analysis"))
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)  # dataset paths are relative to the project root
    block_downloads()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = use_cuda  # same as Level 4.3 so the predictions match exactly

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_relative_to(RESULTS_DIR.resolve()):
        fail(f"output directory must be under {RESULTS_DIR}")
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        fail(f"checkpoint not found: {checkpoint_path}")

    print("=" * 64)
    print("LEVEL 4.4 - ERROR ANALYSIS: pre-checks")
    print("=" * 64)
    sha_before = sha256(checkpoint_path)
    model, checkpoint = load_model(checkpoint_path, device)
    print(f"[PASS] Loaded {checkpoint_path.relative_to(PROJECT_ROOT)} (epoch {checkpoint['epoch']}, "
          f"val loss {checkpoint['metrics']['val_loss']:.4f}) on {device}")

    test_set, test_loader = build_test_loader(use_cuda)
    counts = {CLASS_NAMES[c]: test_set.targets.count(c) for c in CLASS_NAMES}
    if len(test_set) != EXPECTED_TEST_SIZE or set(counts.values()) != {EXPECTED_PER_CLASS}:
        fail(f"unexpected test set: {len(test_set)} images, {counts}")
    print(f"[PASS] Test set {len(test_set):,} images {counts}, no shuffling, no augmentation")

    print(f"\nRunning inference on {len(test_loader)} batches...", flush=True)
    records = predict_all(model, test_loader, test_set, device, use_amp)
    for r in records:
        r["error_type"] = error_type(r)

    unchanged = weights_unchanged(model, checkpoint)
    sha_after = sha256(checkpoint_path)
    if not (unchanged and sha_before == sha_after):
        fail("checkpoint or model weights changed during analysis")
    print("[PASS] Model weights and checkpoint file unchanged")

    # Cross-check against the Level 4.3 evaluation of the same checkpoint
    confusion = [[sum(r["true_label"] == t and r["predicted_label"] == p for r in records)
                  for p in CLASS_NAMES] for t in CLASS_NAMES]
    metrics_file = RESULTS_DIR / f"{args.name}_test_metrics.json"
    if metrics_file.is_file():
        previous = json.loads(metrics_file.read_text(encoding="utf-8"))
        if previous["checkpoint_sha256"] != sha_before:
            print(f"[INFO] {metrics_file.name} was made from a different checkpoint - cross-check skipped")
        elif previous["confusion_matrix"]["values"] != confusion:
            fail(f"confusion {confusion} differs from Level 4.3 {previous['confusion_matrix']['values']}")
        else:
            print(f"[PASS] Confusion matrix identical to Level 4.3 ({metrics_file.name}): {confusion}")

    # Files
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = by_confidence([r for r in records if not r["correct"]])
    groups = {name: [r for r in errors if r["error_type"] == name] for name in ERROR_GROUPS}

    write_csv(output_dir / "all_predictions.csv", records, ranked=False)
    write_csv(output_dir / "errors.csv", errors)  # rank = position among all errors

    sheets = []
    for name, members in groups.items():
        for rank, r in enumerate(members, start=1):
            r["rank"] = rank  # rank within this group, shown on the contact sheets
        write_csv(output_dir / f"{name}s.csv", members)
        g = ERROR_GROUPS[name]
        selections = {
            "highest_confidence": (members[:SHEET_SIZE],
                                   f"{SHEET_SIZE} most confident of {len(members)} ({g['meaning']})"),
            "lowest_confidence": (members[-SHEET_SIZE:],
                                  f"{SHEET_SIZE} least confident of {len(members)} ({g['meaning']})"),
            "representative": (representative(members, SHEET_SIZE),
                               f"{SHEET_SIZE} evenly spaced across all {len(members)} by confidence"),
        }
        for selection, (chosen, subtitle) in selections.items():
            path = output_dir / f"sheet_{name}_{selection}.png"
            contact_sheet(chosen, f"{g['title']} - {selection.replace('_', ' ')}", subtitle, path)
            sheets.append(path)

    checkpoint_info = {
        "path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "epoch": checkpoint["epoch"],
        "validation_loss": checkpoint["metrics"]["val_loss"],
        "sha256": sha_before,
    }
    summary = build_summary(records, checkpoint_info)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = summary_text(summary)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")

    print("\n" + text)
    print("\nFiles:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix():<62} {sha256(path)[:16]}")
    return True


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    sys.exit(0 if main() else 1)
