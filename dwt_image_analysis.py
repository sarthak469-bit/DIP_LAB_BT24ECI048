
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


def perform_dwt(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not find '{image_path}'. "
            "Place an image in the project folder."
        )

    # Convert to floating-point values
    image = image.astype(np.float64) / 255.0

    # Perform one-level Haar wavelet decomposition
    LL, (LH, HL, HH) = pywt.dwt2(image, "haar")

    # Reconstruct the original image
    reconstructed = pywt.idwt2((LL, (LH, HL, HH)), "haar")

    # Match dimensions in case the input has odd dimensions
    reconstructed = reconstructed[:image.shape[0], :image.shape[1]]

    # Calculate reconstruction error
    error = np.max(np.abs(image - reconstructed))
    print(f"Maximum reconstruction error: {error:.10f}")

    # Display the original image and four subbands
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")

    subbands = [
        (LL, "LL - Approximation"),
        (LH, "LH - Detail"),
        (HL, "HL - Detail"),
        (HH, "HH - Detail"),
    ]

    for ax, (band, title) in zip(axes.flat[1:], subbands):
        ax.imshow(band, cmap="gray")
        ax.set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("dwt_results.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    perform_dwt("input.jpg")
