import importlib
import platform
import sys

MIN_PYTHON = (3, 10)
LABEL_WIDTH = 15


def print_line(label, value):
    print(f"{label:<{LABEL_WIDTH}}: {value}")


def check_python():
    version = platform.python_version()
    if sys.version_info < MIN_PYTHON:
        return False, f"{version} (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer)"
    return True, version


def check_package(module_name):
    """Import a package and return (passed, version or error message)."""
    try:
        module = importlib.import_module(module_name)
        return True, module.__version__
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def check_cuda():
    """Return (passed, detail). Never raises, even if PyTorch is missing."""
    try:
        import torch
    except Exception:
        return False, "PyTorch could not be imported"

    if torch.cuda.is_available():
        return True, f"CUDA {torch.version.cuda}"

    if torch.version.cuda is None:
        return False, "PyTorch is a CPU-only build (no CUDA support installed)"
    return False, (
        f"PyTorch has CUDA {torch.version.cuda}, but no usable NVIDIA GPU/driver "
        "was found (check that nvidia-smi works)"
    )


def check_gpu_tensor():
    """Multiply two small matrices on the GPU and check the answer."""
    try:
        import torch

        device = torch.device("cuda")
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)

        result = torch.matmul(a, b)
        torch.cuda.synchronize()  # wait for the GPU to finish

        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        if result.device.type != "cuda":
            return False, "calculation did not run on the GPU"
        if not torch.equal(result.cpu(), expected):
            return False, "GPU returned a wrong result"
        return True, f"2x2 matrix multiply on {result.device}"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def status(passed):
    return "PASS" if passed else "FAIL"


def main():
    print("=" * 50)
    print("DEEPFAKE PROJECT ENVIRONMENT TEST")
    print("=" * 50)
    print()

    # name shown in the report -> import name
    packages = {
        "NumPy": "numpy",
        "OpenCV": "cv2",
        "Pillow": "PIL",
        "Matplotlib": "matplotlib",
        "PyTorch": "torch",
        "Torchvision": "torchvision",
    }

    failures = []

    passed, detail = check_python()
    print_line("Python", f"{status(passed)}  ({detail})")
    if not passed:
        failures.append(("Python", detail))

    for display_name, module_name in packages.items():
        passed, detail = check_package(module_name)
        print_line(display_name, f"{status(passed)}  ({detail})")
        if not passed:
            failures.append((display_name, detail))

    cuda_ok, cuda_detail = check_cuda()
    print_line("CUDA", f"{status(cuda_ok)}  ({cuda_detail})")
    if not cuda_ok:
        failures.append(("CUDA", cuda_detail))

    if cuda_ok:
        import torch

        print_line("GPU", torch.cuda.get_device_name(0))
        tensor_ok, tensor_detail = check_gpu_tensor()
    else:
        print_line("GPU", "not detected")
        tensor_ok, tensor_detail = False, "skipped because CUDA is not available"

    print()
    print_line("GPU Tensor Test", f"{status(tensor_ok)}  ({tensor_detail})")
    if not tensor_ok:
        failures.append(("GPU Tensor Test", tensor_detail))

    print()
    print("=" * 50)
    if failures:
        print("ENVIRONMENT NOT READY")
        print("=" * 50)
        print("\nFailed checks:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        return False

    print("ENVIRONMENT READY")
    print("=" * 50)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
