import numpy as np
import cv2 as cv

im_size = 300


# printing information about image
def print_info(image):
    image = cv.cvtColor(image, cv.CV_8U)
    height, width, lol = image.shape
    print("\npicture type: ", type(image), "\npicture heigth: ", height, "\npicture width: ", width)


def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value


# rotation of a picture 90 degrees clockwise
def rotate_picture(picture, second_picture):
    for j in range(len(picture) // 2):
        for i in range((len(picture) - ((j * 2) + 1))):
            picture[j + i, j] = second_picture[-j - 1, i + j]
            picture[-j - 1, i + j] = second_picture[-j - i - 1, -j - 1]
            picture[-j - i - 1, -j - 1] = second_picture[j, -j - i - 1]
            picture[j, -j - i - 1] = second_picture[j + i, j]
    return picture


def sharpen(my_image):
    my_image = cv.cvtColor(my_image, cv.CV_8U)
    height, width, n_channels = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)

    for j in range(0, (im_size * 2) - 1):
        for i in range(0, (im_size * 2) - 1):
            for k in range(0, n_channels):
                result[j, i, k] = my_image[j, i, k]

    for j in range(1, im_size - 1):
        for i in range(1, im_size - 1):

            for k in range(0, n_channels):
                sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k] \
                            - my_image[j - 1, i, k] - my_image[j, i + 1, k] \
                            - my_image[j, i - 1, k]
                result[j, i, k] = saturated(sum_value)

    return result


def concat_vh(list_2d):
    return cv.vconcat([cv.hconcat(list_h)
                       for list_h in list_2d])


# get image only in red colour
def get_red_channel(image):
    for j in range(len(image)):
        for i in range(len(image)):
            # image[px, py, color] = [px, py, [blue, green, red]] -> setting blue and green channels to zero
            image[i, j, 0] = 0
            image[i, j, 1] = 0


counter = 0
frame = []
# main loop
while cv.waitKey() != ord('q'):
    cap = cv.VideoCapture(0)
    rat1, pom = cap.read()
    cap.release()
    img1 = cv.resize(pom, (im_size, im_size))
    # cv.imwrite("pic1.jpg", img1)
    cv.imshow("test", img1)

    while counter < 4:
        c = cv.waitKey(1)
        if c == ord(' '):
            cap = cv.VideoCapture(0)
            rat1, x = cap.read()
            cap.release()
            frame.append(x)
            counter += 1
            print(counter, ". fotka spravena ")

    # resizing our 4 pictures
    img1 = cv.resize(frame[0], (im_size, im_size))
    img2 = cv.resize(frame[1], (im_size, im_size))
    img3 = cv.resize(frame[2], (im_size, im_size))
    img4 = cv.resize(frame[3], (im_size, im_size))

    cv.imwrite("pic1.jpg", img1)
    cv.imwrite("pic2.jpg", img2)
    cv.imwrite("pic3.jpg", img3)
    cv.imwrite("pic4.jpg", img4)

    # creating 2 by 2 matrix with our 4 images
    img_tile = concat_vh([[img1, img2],
                          [img3, img4]])

    cv.imshow("pociatocna mozaika", img_tile)

    # rotating 2nd picture
    temp = cv.imread('pic2.jpg')
    img_tile[0:im_size, im_size:im_size * 2, :] = rotate_picture(img2, temp)

    # showing 3rd picture only in red colour
    get_red_channel(img3)
    img_tile[im_size:im_size * 2, 0:im_size, :] = img3

    # sharpening 1st picture
    img_tile = sharpen(img_tile)

    # final image
    cv.imshow("upravena mozaika", img_tile)
    print_info(img_tile)