import math
import cv2 as cv
import numpy as np

src = cv.imread('house.jpg')
original = np.copy(src)

gray_image = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
blurred_image = cv.GaussianBlur(gray_image, (9, 9), 0)
edges_image = cv.Canny(blurred_image, 120, 150)

lines = cv.HoughLines(edges_image, 1, np.pi/180, 100, None, 0, 0)
counter = 1
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        print(f"rho: {rho}, theta: {np.rad2deg(theta)}")
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(original, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)
        counter += 1

cv.imshow("Source", src)
cv.imshow("Canny image", edges_image)
cv.imshow("Detected lines", original)
# cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv.waitKey()
