"""
Stage 2 fine-tuning: shared configuration, paths and safety guards.

Stage 2 starts from the Stage 1 best checkpoint and trains the WHOLE DeFaX
model (Swin-T and EfficientNet-B0 unfrozen) with a small learning rate.

Experimental rule: the test split is LOCKED. Stage 2 only ever builds the
train and validation ImageFolders, and TestSplitGuard blocks (raises on) any
file open or directory listing under the test folder while it is active.

Safety rules enforced here:
- training data must come from train/, validation data from valid/
- the validation transform must contain no random operations
- every write must go to the Stage 2 folders below; Stage 1 checkpoints and
  results are refused by name
"""

import hashlib
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.dataset import DATASET_ROOT
from src.data.transforms import train_transform, valid_test_transform
from src.training.trainer import CLASS_NAMES, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (PROJECT_ROOT / DATASET_ROOT).resolve()
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"

STAGE1_BEST = PROJECT_ROOT / "models" / "checkpoints" / "best.pt"
STAGE2_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "stage2"
STAGE2_RESULTS_DIR = PROJECT_ROOT / "results" / "stage2"

# Stage 1 outputs that Stage 2 must never write to
PROTECTED_PATHS = [
    PROJECT_ROOT / "models" / "checkpoints" / "best.pt",
    PROJECT_ROOT / "models" / "checkpoints" / "last.pt",
    PROJECT_ROOT / "models" / "checkpoints" / "defax_smoke_test.pt",
    PROJECT_ROOT / "results" / "stage1_history.csv",
    PROJECT_ROOT / "results" / "stage1_test_evaluation.txt",
    PROJECT_ROOT / "results" / "stage1_test_metrics.json",
    PROJECT_ROOT / "results" / "stage1_test_confusion_matrix.png",
    PROJECT_ROOT / "results" / "error_analysis",
]

STAGE2_CONFIG = {
    "stage": "stage2",
    "seed": 42,
    "init_checkpoint": "models/checkpoints/best.pt",  # Stage 1 best (read only)
    "epochs": 5,                  # initial experimental setting (5-10 planned)
    "batch_size": 8,
    "eval_batch_size": 32,        # no gradients during validation, so larger is fine
    "learning_rate": 1e-5,        # one rate for all parameters (initial setting)
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "min_learning_rate": 1e-7,
    "max_grad_norm": 1.0,         # clip gradient spikes during full fine-tuning
    "freeze_batchnorm": True,     # keep BatchNorm running stats fixed at batch size 8
    "freeze_backbones": False,
    "amp": True,
    "num_workers": 4,
    "log_every": 1000,
}


# ------------------------------------------------------------------ guards

def _normalise(path):
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


class TestSplitGuard:
    """
    Blocks any access to the test split using Python audit hooks.

    While active, opening a file or listing a directory under TEST_DIR raises
    PermissionError, and the attempt is recorded. Audit hooks cannot be
    removed, so the hook is installed once and switched on/off with `active`.
    Covers the current process (use num_workers=0 to cover image loading
    too; with workers, the dataset file lists are checked separately).
    """

    _installed = None

    def __init__(self, test_dir=TEST_DIR):
        self.test_dir = _normalise(test_dir)
        self.active = False
        self.blocked = []

    @classmethod
    def install(cls):
        if cls._installed is None:
            cls._installed = cls()
            sys.addaudithook(cls._installed._hook)
        return cls._installed

    def _hook(self, event, args):
        if not self.active or event not in ("open", "os.scandir", "os.listdir"):
            return
        target = args[0] if args else None
        if not isinstance(target, (str, bytes, os.PathLike)):
            return  # e.g. file descriptors
        path = _normalise(target)
        if path == self.test_dir or path.startswith(self.test_dir + os.sep):
            self.blocked.append(path)
            raise PermissionError(f"Stage 2 test-split guard: access to {path} is blocked")


def assert_stage2_output(path):
    """Refuse to write anywhere except the Stage 2 folders, and never to Stage 1 files."""
    resolved = Path(path).resolve()
    for protected in PROTECTED_PATHS:
        protected = protected.resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise PermissionError(f"Refusing to write to protected Stage 1 output: {resolved}")
    if not (resolved.is_relative_to(STAGE2_CHECKPOINT_DIR.resolve())
            or resolved.is_relative_to(STAGE2_RESULTS_DIR.resolve())):
        raise PermissionError(f"Stage 2 may only write under {STAGE2_CHECKPOINT_DIR} or "
                              f"{STAGE2_RESULTS_DIR}, not {resolved}")
    return resolved


def assert_split(dataset, expected_dir):
    """Every sample must live under expected_dir (train/ or valid/), never test/."""
    expected = _normalise(expected_dir)
    test = _normalise(TEST_DIR)
    for path, _ in dataset.samples:
        p = _normalise(path)
        if p.startswith(test + os.sep) or not p.startswith(expected + os.sep):
            raise RuntimeError(f"Sample outside {expected_dir}: {path}")


def random_operations(transform):
    return [type(t).__name__ for t in transform.transforms if type(t).__name__.startswith("Random")]


def stage1_artifact_files():
    """Every existing Stage 1 file (protected directories expanded), for fingerprinting."""
    return sorted(
        f for p in PROTECTED_PATHS
        for f in ([p] if p.is_file() else (p.iterdir() if p.is_dir() else []))
        if f.is_file()
    )


# ------------------------------------------------------------------ verification helpers

def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(module, before):
    """Number of parameter tensors that differ from a snapshot."""
    return sum(not torch.equal(p.detach(), b) for p, b in zip(module.parameters(), before))


def weights_fingerprint(model):
    """SHA-256 over every tensor in the state dict (bitwise reproducibility checks)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ building blocks

def build_datasets():
    """Train and validation ImageFolders only - the test folder is never listed."""
    if random_operations(valid_test_transform):
        raise RuntimeError(f"Validation transform has random ops: {random_operations(valid_test_transform)}")
    train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    valid_set = datasets.ImageFolder(VALID_DIR, transform=valid_test_transform)
    for ds in (train_set, valid_set):
        if ds.class_to_idx != {"fake": 0, "real": 1}:
            raise RuntimeError(f"Unexpected class_to_idx {ds.class_to_idx}")
    assert_split(train_set, TRAIN_DIR)
    assert_split(valid_set, VALID_DIR)
    return train_set, valid_set


def guard_worker(worker_id):
    """
    DataLoader worker_init_fn: workers are separate processes, so the main
    process's audit hook does not cover them. Install and self-test the
    test-split guard inside every worker; refuse to run if it is not active.
    """
    guard = TestSplitGuard.install()
    guard.active = True
    try:
        os.listdir(TEST_DIR)
    except PermissionError:
        guard.blocked.clear()  # the self-test block is expected
        return
    raise RuntimeError(f"Test-split guard failed to activate in DataLoader worker {worker_id}")


def build_loader(dataset, batch_size, shuffle, num_workers, use_cuda, generator=None,
                 persistent_workers=None):
    """
    persistent_workers defaults to True when workers are used. Pass False for
    the training loader so workers restart (and are re-seeded from
    `generator`) every epoch - an epoch then sees the same order and
    augmentation whether it runs straight through or after --resume.
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=num_workers, pin_memory=use_cuda,
                      persistent_workers=persistent_workers and num_workers > 0,
                      worker_init_fn=guard_worker if num_workers > 0 else None)


def build_model_from_stage1(config, device):
    """
    DeFaX with BOTH backbones trainable, initialised from the Stage 1 best
    checkpoint (read only). Random init + strict load, so nothing downloads.
    """
    from src.models.defax import DeFaX
    checkpoint = torch.load(PROJECT_ROOT / config["init_checkpoint"], map_location="cpu",
                            weights_only=True)
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise RuntimeError(f"Stage 1 label map {checkpoint.get('class_names')} != {CLASS_NAMES}")
    model = DeFaX(pretrained=False, freeze_backbones=config["freeze_backbones"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device), checkpoint


def build_optimizer(model, config):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"])
    return optimizer, scheduler


def save_stage2_checkpoint(path, model, optimizer, scheduler, scaler, epoch, config,
                           history, best_val_loss, best_epoch):
    """save_checkpoint behind the Stage 2 write guard."""
    path = assert_stage2_output(path)
    latest = history[-1] if history else {}
    return save_checkpoint(
        path, model, optimizer, scaler, epoch,
        metrics={k: latest.get(k) for k in ("val_loss", "val_accuracy", "fake_f1", "real_f1")},
        config=config,
        extra={
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "init_checkpoint": config["init_checkpoint"],
        },
    )
