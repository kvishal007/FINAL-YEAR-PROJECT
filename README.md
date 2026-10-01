# Deepfake Detection Platform

An explainable AI-based deepfake facial-image detection platform, inspired by the DeFaX research architecture.

> **Status:** Early development. Only Level 1 (development environment) is complete.
> The deepfake detector itself has **not** been implemented yet.

---

## Project Overview

Deepfake images, meaning synthetic or manipulated faces, are increasingly hard to tell apart from real photographs. This project aims to build a platform that:

1. Classifies a facial image as **real** or **fake**.
2. **Explains** its decision visually by showing which regions of the face influenced the prediction.
3. Makes the detector available through a REST API and a web application.

The design is inspired by the IEEE paper *"DeFaX: A Cross-Attention Fusion Framework for Robust and Explainable Deepfake Detection."* The project is built in eight levels, and each level is completed and verified before the next one begins.

---

## Current Status

| Area | Status |
|------|--------|
| Development environment (Python, PyTorch, CUDA) | ✅ Completed and verified |
| Dataset and preprocessing | ⏳ Planned |
| Deepfake detection model | ⏳ Planned |
| Explainability (Grad-CAM / LIME) | ⏳ Planned |
| REST API | ⏳ Planned |
| Web application | ⏳ Planned |

At this stage the repository contains **environment verification scripts only**. There is no trained model, no dataset, and no detection functionality yet.

---

## Technology Stack

**In use (Level 1):**

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| NumPy | Numerical arrays and matrix operations |
| OpenCV | Image processing (later: frame extraction, face cropping) |
| Pillow | Image loading and saving |
| Matplotlib | Graphs and visualizations (later: training curves, heatmaps) |
| PyTorch | Deep learning framework, with GPU acceleration through CUDA |
| Torchvision | Image transforms and vision utilities for PyTorch |
| Git | Version control |

**Planned for later levels:** a REST API framework, a web frontend, and explainability libraries. These will be added when their level begins.

---

## Level 1 Environment

Verified with `python tests/test_environment.py`. All checks passed.

| Component | Version |
|-----------|---------|
| Python | 3.12.10 |
| NumPy | 2.5.3 |
| OpenCV | 5.0.0 |
| Pillow | 12.3.0 |
| Matplotlib | 3.11.2 |
| PyTorch | 2.14.1+cu126 |
| Torchvision | 0.29.1+cu126 |
| CUDA (PyTorch build) | 12.6 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |

### Setting up the environment (Windows)

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install the exact package versions
#    (the extra index is needed for the CUDA builds of torch/torchvision)
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 4. Verify
python tests\test_environment.py
```

The expected final line of the verification report is `ENVIRONMENT READY`.

### Verification scripts

| Script | What it checks |
|--------|----------------|
| `tests/test_environment.py` | Every Level 1 component, CUDA, and a GPU calculation. This is the main check. |
| `tests/test_numpy.py` | Matrix calculation |
| `tests/test_opencv.py` | Image creation, saved to `results/opencv_test.png` |
| `tests/test_pillow.py` | Image save and reload, saved to `results/pillow_test.jpg` |
| `tests/test_matplotlib.py` | Graph generation, saved to `results/matplotlib_test.png` |
| `tests/test_pytorch.py` | PyTorch version, CUDA, and a tensor calculation on the GPU |

---

## Project Structure

```
deepfake-detection-platform/
├── src/              # Project source code (planned from Level 2)
├── notebooks/        # Jupyter notebooks for experiments
├── tests/            # Environment verification scripts
├── data/             # Local datasets (not committed to Git)
├── models/           # Trained model checkpoints (not committed to Git)
├── results/          # Generated outputs such as images and graphs (not committed to Git)
├── docs/             # Documentation and research notes
├── requirements.txt  # Exact package versions
├── .gitignore        # Files and folders Git should not track
└── README.md
```

---

## Development Roadmap

| Level | Name | Status |
|-------|------|--------|
| 1 | Development Environment | ✅ **COMPLETED** |
| 2 | Dataset & Preprocessing | ⏳ PLANNED |
| 3 | Baseline Deepfake Classifier | ⏳ PLANNED |
| 4 | DeFaX Architecture | ⏳ PLANNED |
| 5 | Explainable AI | ⏳ PLANNED |
| 6 | REST API | ⏳ PLANNED |
| 7 | Web Application | ⏳ PLANNED |
| 8 | Deployment & Commercial Prototype | ⏳ PLANNED |

---

## Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB |
| NVIDIA driver | 566.07 |

So far the GPU has only been used for environment verification, a small tensor calculation. No model has been trained yet. Because the GPU has 4 GB of VRAM, later training will likely need techniques such as small batch sizes and mixed-precision training.

---

## Future Architecture (Planned)

> ⚠️ **This section describes the planned design. None of these components is implemented yet.**

```
            Input facial image
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Swin Transformer         EfficientNet
  (global context)      (fine local detail)
        └───────────┬───────────┘
                    ▼
         Cross-Attention Fusion
                    ▼
     Classification (Real / Fake)
                    ▼
     Explainability (Grad-CAM / LIME)
                    ▼
                REST API
                    ▼
            Web Application
```

- **Swin Transformer** is meant to capture global structure across the face.
- **EfficientNet** is meant to capture fine-grained local artifacts.
- **Cross-attention fusion** is meant to combine the two feature streams.
- **Grad-CAM / LIME** are meant to produce visual explanations of each prediction.

---

## Important Disclaimer

This is an academic final-year project under active development.

- The final detector's performance will be **measured experimentally on selected datasets**, and the results will be reported for those datasets only.
- The project does **not** claim universal deepfake detection. Detection performance may not carry over to unseen generation methods, image qualities, or datasets.
- No accuracy is guaranteed. The platform's predictions should not be the only basis for any legal, journalistic, or security decision.
