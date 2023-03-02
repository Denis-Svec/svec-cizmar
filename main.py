import numpy as np
import cv2 as cv
import time

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value


def sharpen(my_image):

    my_image = cv.cvtColor(my_image, cv.CV_8U)
    height, width, n_channels = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)


    for j in range(0, 600 - 1):
        for i in range(0, 600 - 1):
            for k in range(0, n_channels):
                result[j, i, k] = my_image[j, i, k]

    for j in range(1, 300 - 1):
        for i in range(1, 300 - 1):

            for k in range(0, n_channels):
                sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k] \
                                - my_image[j - 1, i, k] - my_image[j, i + 1, k] \
                                - my_image[j, i - 1, k]
                result[j, i, k] = saturated(sum_value)

    return result


def concat_vh(list_2d):

    return cv.vconcat([cv.hconcat(list_h)
                        for list_h in list_2d])

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


        img_tile = concat_vh([[img1, img2],
                              [img3, img4]])

        dst0 = sharpen(img_tile)

        cv.imshow('panorama.jpg', dst0)
        cap.release()


        cv.waitKey()
