import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image 
img = cv2.imread('input.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1.Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Split into B, G, R channels
B, G, R = cv2.split(img)
zeros = np.zeros_like(B)

# 3. Build "resolution" images — each channel shown with others zeroed out
red_img   = cv2.merge([zeros, zeros, R])
green_img = cv2.merge([zeros, G, zeros])
blue_img  = cv2.merge([B, zeros, zeros])

# Convert back to RGB for correct display with matplotlib
red_img   = cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB)
green_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB)
blue_img  = cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)

# Display all results
titles = ['Original', 'Grayscale', 'Red Channel', 'Green Channel', 'Blue Channel']
images = [img_rgb, gray, red_img, green_img, blue_img]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    if titles[i] == 'Grayscale':
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('output.png')  
plt.show()