import torch


def main():
    print("=" * 50)
    print("PYTORCH GPU TEST")
    print("=" * 50)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"CUDA version    : {torch.version.cuda}")

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

        # Create tensors directly on the GPU
        a = torch.tensor(
            [[1.0, 2.0],
             [3.0, 4.0]],
            device=device
        )

        b = torch.tensor(
            [[5.0, 6.0],
             [7.0, 8.0]],
            device=device
        )

        result = torch.matmul(a, b)

        print("\nTensor A:")
        print(a)

        print("\nTensor B:")
        print(b)

        print("\nA × B:")
        print(result)

        print(f"\nCalculation device: {result.device}")

        if result.device.type == "cuda":
            print("\nGPU TEST: PASS")
        else:
            print("\nGPU TEST: FAIL")

    else:
        print("\nCUDA is not available.")
        print("GPU TEST: FAIL")


if __name__ == "__main__":
    main()