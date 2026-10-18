
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path


# Create an output directory
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# Load image as grayscale
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        "Place input.jpg in the same folder as this script."
    )

image = image.astype(np.float32) / 255.0


# --------------------------------------------------
# PART A: FIRST-ORDER EDGE DETECTION
# --------------------------------------------------

# Roberts cross operator
roberts_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)

roberts_y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

rx = ndimage.convolve(image, roberts_x)
ry = ndimage.convolve(image, roberts_y)
roberts_manual = np.hypot(rx, ry)


# Prewitt operator
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_y = prewitt_x.T

px = ndimage.convolve(image, prewitt_x)
py = ndimage.convolve(image, prewitt_y)
prewitt_manual = np.hypot(px, py)


# Sobel operator
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = sobel_x.T

sx = ndimage.convolve(image, sobel_x)
sy = ndimage.convolve(image, sobel_y)
sobel_manual = np.hypot(sx, sy)


# Built-in Sobel implementation
sobel_built_x = cv2.Sobel(
    image, cv2.CV_32F, 1, 0, ksize=3
)
sobel_built_y = cv2.Sobel(
    image, cv2.CV_32F, 0, 1, ksize=3
)
sobel_builtin = np.hypot(sobel_built_x, sobel_built_y)


# --------------------------------------------------
# PART B: SECOND-ORDER EDGE DETECTION
# --------------------------------------------------

# Smooth first to reduce noise
smoothed = cv2.GaussianBlur(
    image, (0, 0), sigmaX=1.2
)

# Laplacian of Gaussian (LoG)
log_response = cv2.Laplacian(
    smoothed, cv2.CV_32F, ksize=3
)


# Difference of Gaussians (DoG)
blur_small = cv2.GaussianBlur(
    image, (0, 0), sigmaX=1.0
)
blur_large = cv2.GaussianBlur(
    image, (0, 0), sigmaX=2.0
)

dog_response = blur_small - blur_large


# Zero-crossing detector
def detect_zero_crossings(response, threshold=0.02):
    positive = ndimage.maximum_filter(response, size=3)
    negative = ndimage.minimum_filter(response, size=3)

    # A neighborhood crosses zero if it has both signs
    crosses = (positive > 0) & (negative < 0)

    # Reject weak responses to reduce noise
    strong = (positive - negative) > threshold

    return crosses & strong


log_edges = detect_zero_crossings(log_response)
dog_edges = detect_zero_crossings(dog_response)


# --------------------------------------------------
# PART C: DISPLAY AND SAVE RESULTS
# --------------------------------------------------

results = [
    (image, "Original Image"),
    (roberts_manual, "Roberts Operator"),
    (prewitt_manual, "Prewitt Operator"),
    (sobel_manual, "Sobel - Manual"),
    (sobel_builtin, "Sobel - OpenCV"),
    (log_response, "LoG Response"),
    (log_edges, "LoG Zero Crossings"),
    (dog_response, "DoG Response"),
    (dog_edges, "DoG Zero Crossings"),
]

fig, axes = plt.subplots(3, 3, figsize=(13, 12))

for ax, (result, title) in zip(axes.flat, results):
    ax.imshow(result, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT / "edge_detection_results.png", dpi=200)
plt.show()

print("Edge detection completed.")
print("Results saved to outputs/edge_detection_results.png")
