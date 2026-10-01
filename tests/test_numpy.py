import sys


def main():
    print("=" * 50)
    print("NUMPY TEST")
    print("=" * 50)

    try:
        import numpy as np

        print(f"NumPy version : {np.__version__}")

        # A small 2x2 matrix
        matrix = np.array(
            [[1, 2],
             [3, 4]]
        )

        # Simple numerical operations
        doubled = matrix * 2
        product = matrix @ matrix  # matrix multiplication

        print("\nOriginal matrix:")
        print(matrix)

        print("\nMatrix × 2:")
        print(doubled)

        print("\nMatrix × Matrix:")
        print(product)

        # Check the answer is what we expect
        expected = np.array(
            [[7, 10],
             [15, 22]]
        )
        if not np.array_equal(product, expected):
            raise ValueError("Matrix multiplication gave an unexpected result")

        print("\nNUMPY TEST: PASS")
        return True

    except Exception as error:
        print(f"\nError: {error}")
        print("NUMPY TEST: FAIL")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
