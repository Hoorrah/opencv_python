import cv2
import numpy as np
from matplotlib import pyplot as plt


def empty(a):
	pass

#cv2.namedWindow("Parameters")
#cv2.resizeWindow("Parameters", 640, 240)
#cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def getContours(img, imgContour):

	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL: only extreme outer contours , CHAIN_APPROX_NONE: we use non to get all the points
	#cv2.drawContours(imgContour, contours, -1, (0,0,0), 5)

	for cnt in contours:
		area = cv2.contourArea(cnt)
		#areaMin =cv2.getTrackbarPos("Area", "Parameters")
		if area > 1000:
			cv2.drawContours(imgContour, cnt, -1, (0,0,0), 5)


img = cv2.imread('30_cover.jpg')
imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (15,15), 1)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = auto_canny(imgGray)
kernel = np.ones((3,3))
imgDil = cv2.dilate(imgCanny, kernel, iterations =1)

getContours(imgDil, imgContour)
cv2.imshow('imaged', imgDil)
cv2.imshow('image', imgContour)
#cv.imshow('image gray', imgray)
cv2.waitKey()
cv2.destroyAllWindows()
