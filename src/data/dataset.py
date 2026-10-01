from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from transforms import train_transform, valid_test_transform


DATASET_ROOT = Path(
    "data/real_vs_fake/real-vs-fake"
)

BATCH_SIZE = 32
NUM_WORKERS = 0


def create_datasets():
    train_dataset = datasets.ImageFolder(
        DATASET_ROOT / "train",
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        DATASET_ROOT / "valid",
        transform=valid_test_transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_ROOT / "test",
        transform=valid_test_transform
    )

    return train_dataset, valid_dataset, test_dataset


def create_dataloaders():
    train_dataset, valid_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = create_datasets()

    print("=" * 60)
    print("DEEPFAKE DATASET")
    print("=" * 60)

    print(f"Train images      : {len(train_dataset)}")
    print(f"Validation images : {len(valid_dataset)}")
    print(f"Test images       : {len(test_dataset)}")

    print("\nClass mapping:")
    print(train_dataset.class_to_idx)

    train_loader, valid_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"Image tensor shape : {images.shape}")
    print(f"Label tensor shape : {labels.shape}")
    print(f"Image data type    : {images.dtype}")
    print(f"Label data type    : {labels.dtype}")

    print("\n" + "=" * 60)
    print("DATA PIPELINE TEST COMPLETE")
    print("=" * 60)