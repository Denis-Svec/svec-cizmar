import numpy as np
from scipy import ndimage
import cv2
import numpy as np
from scipy import ndimage


def canny(image, sigma=1, low_threshold=10, high_threshold=20):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Apply Gaussian smoothing to the image
    smoothed = ndimage.gaussian_filter(image, sigma)

    # Compute the gradient magnitude and direction
    dx, dy = np.gradient(smoothed)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx) * 180 / np.pi

    # Perform non-maximum suppression
    theta[theta < 0] += 180
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if (theta[i, j] >= 0 and theta[i, j] < 22.5) or (theta[i, j] >= 157.5 and theta[i, j] < 180):
                q = magnitude[i, j + 1] if j + 1 < cols else 0
                r = magnitude[i, j - 1] if j - 1 >= 0 else 0
            elif (theta[i, j] >= 22.5 and theta[i, j] < 67.5):
                q = magnitude[i + 1, j + 1] if i + 1 < rows and j + 1 < cols else 0
                r = magnitude[i - 1, j - 1] if i - 1 >= 0 and j - 1 >= 0 else 0
            elif (theta[i, j] >= 67.5 and theta[i, j] < 112.5):
                q = magnitude[i + 1, j] if i + 1 < rows else 0
                r = magnitude[i - 1, j] if i - 1 >= 0 else 0
            else:
                q = magnitude[i - 1, j + 1] if i - 1 >= 0 and j + 1 < cols else 0
                r = magnitude[i + 1, j - 1] if i + 1 < rows and j - 1 >= 0 else 0
            if magnitude[i, j] < q or magnitude[i, j] < r:
                magnitude[i, j] = 0

    # Perform hysteresis thresholding
    thresholded = np.zeros_like(magnitude)
    high_pixels = magnitude > high_threshold
    thresholded[high_pixels] = 255
    low_pixels = (magnitude >= low_threshold) & (magnitude <= high_threshold)
    labels, count = ndimage.label(low_pixels)
    for i in range(1, count + 1):
        label_mask = np.where(labels == i, 255, 0)
        connected_pixels = np.where(high_pixels & ndimage.binary_dilation(label_mask), 255, 0)
        thresholded = np.maximum(thresholded, connected_pixels)

    return thresholded
if __name__ == '__main__':


    image = cv2.imread('pic1.jpg')


    done = canny(image)
    cv2.imshow('concat_vh.jpg', done)
    cv2.waitKey()