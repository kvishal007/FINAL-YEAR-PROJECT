import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "results" / "matplotlib_test.png"


def main():
    print("=" * 50)
    print("MATPLOTLIB TEST")
    print("=" * 50)

    try:
        import matplotlib

        # "Agg" draws straight to a file, so no window ever opens
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print(f"Matplotlib version : {matplotlib.__version__}")
        print(f"Backend            : {matplotlib.get_backend()}")

        # Example data shaped like a training loss curve
        epochs = [1, 2, 3, 4, 5]
        loss = [0.90, 0.65, 0.45, 0.33, 0.25]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, loss, marker="o")
        ax.set_title("Matplotlib Test Graph")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FILE, dpi=100)
        plt.close(fig)

        if not OUTPUT_FILE.exists():
            raise FileNotFoundError(f"Graph was not created: {OUTPUT_FILE}")

        print(f"Saved to           : {OUTPUT_FILE}")
        print(f"File size          : {OUTPUT_FILE.stat().st_size} bytes")

        print("\nMATPLOTLIB TEST: PASS")
        return True

    except Exception as error:
        print(f"\nError: {error}")
        print("MATPLOTLIB TEST: FAIL")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
