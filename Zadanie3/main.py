import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

PI = math.pi
source = cv.imread('house.jpg')
original = np.copy(source)
threshold = 110


def normalise(picture, x_rows, y_cols):
    for i in range(0, x_rows):
        for j in range(0, y_cols):
            if picture[i, j] > 245:
                picture[i, j] = 255
            else:
                picture[i, j] = 0


gray_image = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
blurred_image = cv.GaussianBlur(gray_image, (9, 9), 0)
edge_image = cv.Canny(blurred_image, 100, 150)
copy = np.copy(edge_image)
rows, cols = copy.shape
normalise(copy, rows, cols)

max_distance = int(math.sqrt(rows ** 2 + cols ** 2))
Accumulator = np.zeros((max_distance * 2, 180), dtype=int)
print(f"rows: {rows}, cols: {cols}, max distance: {max_distance}")

angles = np.arange(0, 180, 1)
for x in range(0, rows):
    for y in range(0, cols):
        if copy[x, y] == 255:
            rhos = []
            for theta in angles:
                res = int(x * math.cos(np.deg2rad(theta)) + y * math.sin(np.deg2rad(theta)) + rows)
                rhos.append(res)
                Accumulator[res, theta] += 1
            # plt.plot(angles, rhos)

for x in range(Accumulator.shape[1]):
    for y in range(Accumulator.shape[0]):
        if Accumulator[y, x] >= threshold:
            # print(f"thresholds x: {x} + y: {y}")
            rho = y - rows
            theta = (PI/2) - np.deg2rad(x)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            print(f"rho: {rho}, theta: {np.rad2deg(theta)}")
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv.line(original, pt1, pt2, (255, 255, 0), 1, cv.LINE_AA)

# Function to show the heat map

plt.imshow(Accumulator, cmap='magma')

# Adding details to the plot
plt.title("2-D Heat Map")
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Adding a color bar to the plot
plt.colorbar()
plt.savefig('accumulator.png')
# Displaying the plot
plt.show()

cv.imshow("Source", source)
cv.imshow("Canny image", edge_image)
cv.imshow("Detected lines", original)
# cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv.waitKey()
