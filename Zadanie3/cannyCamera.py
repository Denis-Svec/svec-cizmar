import cv2 as cv

cap = cv.VideoCapture(0)

while cv.waitKey(1) != ord('q'):

    rat1, frame1 = cap.read()

    img1 = cv.resize(frame1, (800, 600))

    gray_image = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (9, 9), 0)
    edges_image = cv.Canny(blurred_image, 50, 100)

    cv.imshow('image', edges_image)

    # cv.waitKey()


