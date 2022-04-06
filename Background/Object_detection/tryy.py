import cv2
import numpy as np
from PIL import Image
import os, os.path
import glob
from matplotlib import pyplot as plt
from patchify import patchify

#calculating the Mean square error for finding the difference between colours
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /=float(imageA.shape[0] * imageA.shape[1])
	return err


def auto_canny(image, sigma=0.00001):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
#define variables
patch_size = 20
num_image = 0


def getContours(img, imgContour):

	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL: only extreme outer contours , CHAIN_APPROX_NONE: we use non to get all the points
	#cv2.drawContours(imgContour, contours, -1, (0,0,0), 5)

	for cnt in contours:
		area = cv2.contourArea(cnt)
		#areaMin =cv2.getTrackbarPos("Area", "Parameters")
		if area > 1000:
			cv2.drawContours(imgContour, cnt, -1, (0,0,0), 5)


input_path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image-2'
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(input_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	img = cv2.imread(os.path.join(input_path,f))
	imgContour = img.copy()
	num_image = num_image + 1
	check_hor = True

	patches_img = patchify(img, (patch_size, patch_size, 3), step=patch_size)

	#finding the color of fisrt left top patch in the image
	prev_patch_img = patches_img[0, 1, 0, :, :, :]
	prev_average_color_row = np.average(prev_patch_img, axis=0)
	prev_average_color = np.average(prev_average_color_row, axis=0)
	prev_d_img = np.ones((312,312,3), dtype=np.uint8)
	prev_d_img[:,:] = prev_average_color
	(prev_B, prev_G, prev_R)= cv2.split(prev_d_img)

	single_patch_img = patches_img[0, patches_img.shape[1]-1 , 0, :, :, :]
	average_color_row = np.average(single_patch_img, axis=0)
	average_color = np.average(average_color_row, axis=0)
	d_img = np.ones((312,312,3), dtype=np.uint8)
	d_img[:,:] = average_color
	(B, G, R)= cv2.split(d_img)

	if ( mse(prev_B , B) <= 300 and mse(prev_G , G) <= 300 and mse(prev_R , R) <= 300):
		check_hor = False
	if not (check_hor):
		imgBlur = cv2.GaussianBlur(img, (3,3), 1)
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imgCanny = auto_canny(imgGray)
		lines = cv2.HoughLinesP(imgCanny,5,np.pi/180,15,np.array([]), 100,10)#src=img, rho=1,theta=pi/180, threshold=100
		for line in lines:
			x1,y1,x2,y2 = line[0]# in this method it gives us 2 coordinate so we don't need to calculate them
			cv2.line(imgContour,(x1,y1),(x2,y2),(0,0,0),5) #src=img, point1 , point 2, color ,thickness

		#getContours(imgDil, imgContour)
		cv2.imwrite('conimage_' + '_'+ str(num_image)+'.jpg', imgContour)
		num_image = num_image+1

		#cv2.imwrite('goodimage_' + '_'+ str(num_image)+'.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
