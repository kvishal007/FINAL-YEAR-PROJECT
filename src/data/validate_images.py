from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("data/real_vs_fake/real-vs-fake")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

total = 0
valid = 0
corrupted = 0

split_counts = {
    "train": {"real": 0, "fake": 0},
    "valid": {"real": 0, "fake": 0},
    "test": {"real": 0, "fake": 0},
}

print("=" * 60)
print("DEEPFAKE DATASET IMAGE VALIDATION")
print("=" * 60)

for split in ["train", "valid", "test"]:
    for class_name in ["real", "fake"]:

        folder = DATASET_ROOT / split / class_name

        if not folder.exists():
            print(f"[ERROR] Missing folder: {folder}")
            continue

        files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        for file_path in files:
            total += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()

                valid += 1
                split_counts[split][class_name] += 1

            except Exception:
                corrupted += 1
                print(f"[CORRUPTED] {file_path}")

print("\n" + "=" * 60)
print("VALIDATION RESULT")
print("=" * 60)

print(f"Total images checked : {total}")
print(f"Valid images         : {valid}")
print(f"Corrupted images     : {corrupted}")

print("\nDataset distribution:")
print("-" * 40)

for split, classes in split_counts.items():
    print(
        f"{split:5} | "
        f"Real: {classes['real']:6} | "
        f"Fake: {classes['fake']:6}"
    )

print("-" * 40)

if corrupted == 0:
    print("\nSTATUS: PASS")
    print("All images passed integrity validation.")
else:
    print("\nSTATUS: WARNING")
    print(f"{corrupted} corrupted/unreadable images found.")

print("=" * 60)