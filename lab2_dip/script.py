"""
Histogram Equalization Techniques: GHE, AHE, CLAHE, and Local HE
-------------------------------------------------------------------
Takes one input image and produces four outputs:
 1. Global Histogram Equalization (GHE)
 2. Adaptive Histogram Equalization (AHE) - tile-based, no contrast limiting
 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
 4. Local Histogram Equalization (LHE) - true sliding-window, pixel-by-pixel

Usage:
    python histogram_equalization.py <input_image_path>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from skimage.filters import rank
from skimage.morphology import disk


def global_histogram_equalization(gray_img):
    """1. GHE - equalizes the entire image using one histogram/CDF."""
    return cv2.equalizeHist(gray_img)


def adaptive_histogram_equalization(gray_img, tile_size=(8, 8)):
    """
    2. AHE - equalizes small tiles independently (no clip limit).
    OpenCV's createCLAHE with clipLimit set very high effectively
    disables clipping, giving plain AHE behavior.
    """
    ahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=tile_size)
    return ahe.apply(gray_img)


def clahe_equalization(gray_img, clip_limit=2.0, tile_size=(8, 8)):
    """3. CLAHE - adaptive equalization with contrast clipping to limit noise."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def local_histogram_equalization(gray_img, radius=30):
    """
    4. LHE - true LOCAL histogram equalization.
    Unlike AHE/CLAHE (which equalize fixed non-overlapping tiles),
    this recomputes the histogram in a sliding circular neighborhood
    (radius = 'disk' size) around EVERY pixel and remaps that pixel
    based on its own local neighborhood's CDF.

    Note: computationally heavier than tile-based methods since the
    neighborhood window slides pixel-by-pixel across the whole image.
    """
    footprint = disk(radius)
    result = rank.equalize(gray_img, footprint=footprint)
    return result


def process_image(input_path, output_dir="."):
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")

    # Convert to grayscale for processing
    # (For color images, we equalize only the luminance (Y) channel
    #  to avoid distorting colors, then convert back to BGR.)
    is_color = len(img.shape) == 3 and img.shape[2] == 3

    if is_color:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y_ghe = global_histogram_equalization(y)
        y_ahe = adaptive_histogram_equalization(y)
        y_clahe = clahe_equalization(y)
        y_lhe = local_histogram_equalization(y)

        ghe_img = cv2.cvtColor(cv2.merge([y_ghe, cr, cb]), cv2.COLOR_YCrCb2BGR)
        ahe_img = cv2.cvtColor(cv2.merge([y_ahe, cr, cb]), cv2.COLOR_YCrCb2BGR)
        clahe_img = cv2.cvtColor(cv2.merge([y_clahe, cr, cb]), cv2.COLOR_YCrCb2BGR)
        lhe_img = cv2.cvtColor(cv2.merge([y_lhe, cr, cb]), cv2.COLOR_YCrCb2BGR)
        original_display = img
        gray_for_hist = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ghe_img = global_histogram_equalization(gray)
        ahe_img = adaptive_histogram_equalization(gray)
        clahe_img = clahe_equalization(gray)
        lhe_img = local_histogram_equalization(gray)
        original_display = gray
        gray_for_hist = gray

    os.makedirs(output_dir, exist_ok=True)
    base = "output"

    cv2.imwrite(f"{output_dir}/{base}_1_GHE.png", ghe_img)
    cv2.imwrite(f"{output_dir}/{base}_2_AHE.png", ahe_img)
    cv2.imwrite(f"{output_dir}/{base}_3_CLAHE.png", clahe_img)
    cv2.imwrite(f"{output_dir}/{base}_4_LHE.png", lhe_img)

    # ---- Visualization: images + histograms ----
    results = {
        "Original": (original_display, gray_for_hist),
        "1. Global HE": (ghe_img, cv2.cvtColor(ghe_img, cv2.COLOR_BGR2GRAY) if is_color else ghe_img),
        "2. Adaptive HE": (ahe_img, cv2.cvtColor(ahe_img, cv2.COLOR_BGR2GRAY) if is_color else ahe_img),
        "3. CLAHE": (clahe_img, cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY) if is_color else clahe_img),
        "4. Local HE": (lhe_img, cv2.cvtColor(lhe_img, cv2.COLOR_BGR2GRAY) if is_color else lhe_img),
    }

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    for i, (title, (disp_img, hist_src)) in enumerate(results.items()):
        # Top row: images
        ax_img = axes[0, i]
        if is_color:
            ax_img.imshow(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))
        else:
            ax_img.imshow(disp_img, cmap="gray")
        ax_img.set_title(title)
        ax_img.axis("off")

        # Bottom row: histograms
        ax_hist = axes[1, i]
        ax_hist.hist(hist_src.ravel(), bins=256, range=(0, 256), color="steelblue")
        ax_hist.set_xlim([0, 256])
        ax_hist.set_title(f"{title} Histogram", fontsize=9)

    plt.tight_layout()
    comparison_path = f"{output_dir}/{base}_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    print(f"Saved: {output_dir}/{base}_1_GHE.png")
    print(f"Saved: {output_dir}/{base}_2_AHE.png")
    print(f"Saved: {output_dir}/{base}_3_CLAHE.png")
    print(f"Saved: {output_dir}/{base}_4_LHE.png")
    print(f"Saved comparison figure: {comparison_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python histogram_equalization.py <input_image_path>")
        sys.exit(1)
    process_image(sys.argv[1])