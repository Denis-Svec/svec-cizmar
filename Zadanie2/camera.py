import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 5), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 5), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
cv.destroyAllWindows()

######### CALIBRATION ############################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\nCamera calibrated:\n", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nfx: {}".format(cameraMatrix[0][0]))
print("fy: {}".format(cameraMatrix[1][1]))
print("cx: {}".format(cameraMatrix[0][2]))
print("cy: {}".format(cameraMatrix[1][2]))
print("\nDistorted Parameters:\n", dist)
# print("\nRotation Vectors:\n", rvecs)
# print("\nTranslation Vectors:\n", tvecs)


######### UN-DISTORTION ############################################

img = cv.imread('Chessboard16.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('CalibrationResult.png', dst)

# projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("\ntotal error: {}".format(mean_error / len(objpoints)))

# showing the results
img1 = cv.imread('CalibrationResult.png')
img2 = cv.imread('Chessboard16.jpg')
cv.imshow('CalibrationResult.png', img1)
cv.imshow('Chessboard16.jpg', img2)
cv.waitKey()
