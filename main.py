import numpy as np
import cv2 as cv
import time


while cv.waitKey() != ord('q'):

        cap = cv.VideoCapture(0)

        rat1, frame1 = cap.read()
        cap.release()
        time.sleep(1)
        cap = cv.VideoCapture(0)
        rat2, frame2 = cap.read()
        cap.release()
        time.sleep(1)
        cap = cv.VideoCapture(0)
        rat3, frame3 = cap.read()
        cap.release()
        time.sleep(1)
        cap = cv.VideoCapture(0)
        rat4, frame4 = cap.read()

        img1 = cv.resize(frame1, (300, 300))
        img2 = cv.resize(frame2, (300, 300))
        img3 = cv.resize(frame3, (300, 300))
        img4 = cv.resize(frame4, (300, 300))

        cv.imshow("test", img1)
        cv.imwrite("pic1.jpg", img1)
        cv.imwrite("pic2.jpg", img2)
        cv.imwrite("pic3.jpg", img3)
        cv.imwrite("pic4.jpg", img4)

        cap.release()


        cv.waitKey()
