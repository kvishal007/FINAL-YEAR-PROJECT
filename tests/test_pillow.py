import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "results" / "pillow_test.jpg"


def main():
    print("=" * 50)
    print("PILLOW TEST")
    print("=" * 50)

    try:
        import PIL
        from PIL import Image, ImageDraw

        print(f"Pillow version : {PIL.__version__}")

        # A 300x200 RGB image with a blue background
        width, height = 300, 200
        image = Image.new("RGB", (width, height), color=(30, 90, 200))

        # Draw a white rectangle on it
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 250, 150], outline=(255, 255, 255), width=4)

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        image.save(OUTPUT_FILE)

        if not OUTPUT_FILE.exists():
            raise FileNotFoundError(f"Image was not created: {OUTPUT_FILE}")

        # Reopen the saved file to prove it can be read back
        with Image.open(OUTPUT_FILE) as reloaded:
            print(f"Saved to       : {OUTPUT_FILE}")
            print(f"Dimensions     : {reloaded.width} x {reloaded.height}")
            print(f"Mode           : {reloaded.mode}")

            if reloaded.size != (width, height):
                raise ValueError("Reloaded image has the wrong size")

        print("\nPILLOW TEST: PASS")
        return True

    except Exception as error:
        print(f"\nError: {error}")
        print("PILLOW TEST: FAIL")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
