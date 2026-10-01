import sys
from pathlib import Path

# Project root = the folder above tests/, so the script works from any location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "results" / "opencv_test.png"


def main():
    print("=" * 50)
    print("OPENCV TEST")
    print("=" * 50)

    try:
        import cv2
        import numpy as np

        print(f"OpenCV version : {cv2.__version__}")

        # A 200x300 black image (height, width, 3 colour channels)
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        # OpenCV uses BGR colour order, not RGB
        cv2.rectangle(image, (30, 30), (270, 170), (0, 255, 0), 3)  # green box
        cv2.circle(image, (150, 100), 40, (0, 0, 255), -1)          # filled red circle

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        saved = cv2.imwrite(str(OUTPUT_FILE), image)
        if not saved or not OUTPUT_FILE.exists():
            raise FileNotFoundError(f"Image was not created: {OUTPUT_FILE}")

        print(f"Image shape    : {image.shape}")
        print(f"Saved to       : {OUTPUT_FILE}")
        print(f"File size      : {OUTPUT_FILE.stat().st_size} bytes")

        print("\nOPENCV TEST: PASS")
        return True

    except Exception as error:
        print(f"\nError: {error}")
        print("OPENCV TEST: FAIL")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
