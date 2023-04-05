import numpy as np
import cv2 as cv
import math
from scipy import ndimage

gaussian_kernel = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def normalize(picture):
    x_rows, y_cols = picture.shape
    for i in range(x_rows):
        for j in range(y_cols):
            if picture[i, j] > 245:
                picture[i, j] = 255
            else:
                picture[i, j] = 0

def convolution(input_array, mask):
    input_rows, input_cols = input_array.shape
    picture = np.copy(input_array) * 0
    input_rows -= 1
    input_cols -= 1
    for j in range(1, input_cols):
        for i in range(1, input_rows):
            if j < input_cols or i < input_rows:
                summary = input_array[i - 1, j - 1] * mask[0, 0] + \
                          input_array[i - 1, j] * mask[0, 1] + \
                          input_array[i - 1, j + 1] * mask[0, 2] + \
                          input_array[i, j - 1] * mask[1, 0] + \
                          input_array[i, j] * mask[1, 1] + \
                          input_array[i, j + 1] * mask[1, 2] + \
                          input_array[i + 1, j - 1] * mask[2, 0] + \
                          input_array[i + 1, j] * mask[2, 1] + \
                          input_array[i + 1, j + 1] * mask[2, 2]
                picture[i, j] = summary

    return picture


def canny(image, sigma=1, low_threshold=10, high_threshold=20):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # # Convert the image to grayscale
    # if len(image.shape) == 3:
    #     image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Apply Gaussian smoothing to the image
    smoothed = convolution(image, gaussian_kernel)
    # smoothed = convolution(image, gaussian_kernel)


    # Compute the gradient magnitude and direction
    dx, dy = np.gradient(smoothed)
    # dx = convolution(smoothed, Gx)
    # dy = convolution(smoothed, Gy)
    # normalize(dx)
    # normalize(dy)
    # cv.imshow('dx', dx)
    # cv.imshow('dy', dy)
    # cv.imshow('smoothed', smoothed)
    # cv.waitKey()
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


def edge_detection(image, origin):
    threshold = 125
    PI = math.pi

    copy = np.copy(image)
    rows, cols = copy.shape
    normalize(copy)

    max_distance = int(math.sqrt(rows ** 2 + cols ** 2))
    Accumulator = np.zeros((max_distance * 2, 180), dtype=int)
    print(f"edge detection: \nrows: {rows}, cols: {cols}, max distance: {max_distance}")

    angles = np.arange(0, 180, 1)
    for x in range(0, rows):
        for y in range(0, cols):
            if copy[x, y] == 255:
                rhos = []
                for theta in angles:
                    res = int(x * math.cos(np.deg2rad(theta)) + y * math.sin(np.deg2rad(theta)) + rows)
                    rhos.append(res)
                    Accumulator[res, theta] += 1

    for x in range(Accumulator.shape[1]):
        for y in range(Accumulator.shape[0]):
            if Accumulator[y, x] >= threshold:
                # print(f"thresholds x: {x} + y: {y}")
                rho = y - rows
                theta = (PI / 2) - np.deg2rad(x)
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                print(f"rho: {rho}, theta: {np.rad2deg(theta)}")
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv.line(origin, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)


if __name__ == '__main__':
    source = cv.imread('house.jpg')
    source = cv.resize(source, (600, 480))

    original = np.copy(source)
    canny_image = canny(source)
    edge_detection(canny_image, original)


    cv.imshow("Source", source)
    cv.imshow("Canny image", canny_image)
    cv.imshow("Detected lines", original)
    cv.waitKey()
