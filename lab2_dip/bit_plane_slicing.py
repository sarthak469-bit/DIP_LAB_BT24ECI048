"""
Bit Plane Slicing
-------------------
Decomposes an 8-bit grayscale image into its 8 individual bit planes
(Bit 7 = MSB = most structural info, Bit 0 = LSB = least significant/noise-like).

Usage:
    python bit_plane_slicing.py <input_image_path>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def get_bit_planes(gray_img):
    """
    Extracts all 8 bit planes from a grayscale image.
    Returns a list of 8 binary images (as 0/255 uint8), indexed 0 (LSB) to 7 (MSB).
    """
    bit_planes = []
    for i in range(8):
        plane = (gray_img >> i) & 1     # extract bit i from every pixel
        plane = plane * 255             # scale 0/1 -> 0/255 for visibility
        bit_planes.append(plane.astype(np.uint8))
    return bit_planes


def process_image(input_path, output_dir="."):
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")

    # Convert to grayscale if needed (bit plane slicing works on single-channel intensity)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    bit_planes = get_bit_planes(gray)

    os.makedirs(output_dir, exist_ok=True)
    base = "output"

    # Save each bit plane individually (Bit 7 = MSB ... Bit 0 = LSB)
    for i in range(7, -1, -1):
        cv2.imwrite(f"{output_dir}/{base}_bitplane_{i}.png", bit_planes[i])

    # ---- Visualization: original + all 8 bit planes in a grid ----
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original (Grayscale)")
    axes[0].axis("off")

    for idx, bit in enumerate(range(7, -1, -1)):  # show MSB first
        ax = axes[idx + 1]
        ax.imshow(bit_planes[bit], cmap="gray")
        label = "MSB" if bit == 7 else ("LSB" if bit == 0 else "")
        title = f"Bit {bit}" + (f" ({label})" if label else "")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    comparison_path = f"{output_dir}/{base}_bitplanes_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    for i in range(7, -1, -1):
        print(f"Saved: {output_dir}/{base}_bitplane_{i}.png")
    print(f"Saved comparison figure: {comparison_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bit_plane_slicing.py <input_image_path>")
        sys.exit(1)
    process_image(sys.argv[1])