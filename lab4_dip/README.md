
# Lab 4: Edge Detection in Digital Image Processing

## Aim
To study and implement first-order and second-order edge detection techniques using Python.

## Objectives
- Understand the concept of edge detection in digital images.
- Implement Roberts, Prewitt, and Sobel edge detectors.
- Study second-order edge detection using Laplacian of Gaussian (LoG).
- Implement Difference of Gaussians (DoG).
- Detect edges using the zero-crossing method.
- Compare the results of different edge detection techniques.

## Theory

Edge detection is an image processing technique used to identify boundaries and regions of significant intensity change in an image. It is widely used in image segmentation, object recognition, feature extraction, and computer vision.

### 1. First-Order Edge Detection

First-order edge detection uses image intensity gradients to identify edges.

#### Roberts Operator
The Roberts operator uses two 2 × 2 convolution kernels to detect diagonal intensity changes. It is computationally simple but sensitive to noise.

#### Prewitt Operator
The Prewitt operator uses 3 × 3 horizontal and vertical kernels to estimate image gradients and identify edges.

#### Sobel Operator
The Sobel operator uses 3 × 3 kernels to calculate horizontal and vertical gradients. It gives greater weight to central pixels and provides some smoothing.

The gradient magnitude is calculated as:

G = sqrt(Gx² + Gy²)

where Gx and Gy represent the horizontal and vertical intensity gradients.

### 2. Second-Order Edge Detection

Second-order techniques analyze changes in the image gradient using second derivatives or differences between smoothed images.

#### Laplacian of Gaussian (LoG)
LoG first smooths an image using a Gaussian filter to reduce noise and then applies the Laplacian operator. Edges can be identified from zero crossings in the resulting response.

#### Difference of Gaussians (DoG)
DoG subtracts two Gaussian-blurred versions of an image with different standard deviations. It highlights image structures at particular spatial scales.

#### Zero-Crossing Detection
Zero-crossing detection identifies locations where a response changes sign from positive to negative or vice versa. A threshold can be applied to reduce weak responses caused by noise.

## Technologies Used
- Python 3
- NumPy
- OpenCV
- Matplotlib
- SciPy

## Installation

Install the required libraries using:

```bash
pip install numpy opencv-python matplotlib scipy
```

## Project Structure

```text
lab4_dip/
├── README.md
├── edge_detection.py
├── input.jpg
└── outputs/
    └── edge_detection_results.png
```

## How to Run

1. Place the input image named `input.jpg` in the project directory.
2. Install all required Python libraries.
3. Open a terminal in the `lab4_dip` directory.
4. Run the program:

```bash
python edge_detection.py
```

5. View the displayed edge detection results.
6. The output figure is saved as:

```text
outputs/edge_detection_results.png
```

## Results and Observations

The program displays the original grayscale image and the outputs of different edge detection techniques.

- **Roberts:** Detects diagonal intensity changes using small convolution kernels.
- **Prewitt:** Detects edges using horizontal and vertical gradients.
- **Sobel:** Detects prominent edges using weighted gradient kernels.
- **LoG:** Produces a second-derivative response after Gaussian smoothing.
- **DoG:** Highlights structures by subtracting images blurred at different scales.
- **Zero Crossing:** Identifies sign changes in the LoG and DoG responses.

The appearance of detected edges depends on the input image, smoothing parameters, and threshold values.

## Applications
- Image segmentation
- Object boundary detection
- Feature extraction
- Medical image analysis
- Computer vision
- Object recognition

## Conclusion

This experiment demonstrates the implementation of first-order and second-order edge detection techniques using Python. The Roberts, Prewitt, and Sobel operators detect edges through image gradients, while LoG, DoG, and zero-crossing methods identify edges using second-order responses and scale-based differences.

The experiment helps compare the behavior of different edge detection methods and understand their applications in digital image processing.
```

**Before committing:** make sure the `outputs/edge_detection_results.png` file exists. It is generated when you run `edge_detection.py`.

Suggested commit message:

`Add Lab 4 edge detection README and documentation`

