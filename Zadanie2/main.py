import numpy as np
import cv2 as cv
import time


while cv.waitKey() != ord('q'):


        img1 = cv.imread('circle.jpg')
        gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

        cv.imwrite("pic1.jpg", gray)
        gray = cv.medianBlur(gray, 5)

        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=5)

        if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv.circle(img1, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv.circle(img1, center, radius, (255, 0, 255), 3)

        temp = cv.imread('pic1.jpg')
        cv.imshow("detected circles", img1)




        cv.waitKey()
