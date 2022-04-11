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


def auto_canny(image, sigma=0.05):
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

	for cnt in contours:
		area = cv2.contourArea(cnt)
		#areaMin =cv2.getTrackbarPos("Area", "Parameters")
		if area > 1000:
			cv2.drawContours(imgContour, cnt, -1, (0,0,0), 5)


def crop_vertically(img, imgContour, num_image, patch_size_h, limit): #1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mse
	loc_right_left =[]
	#define variables
	loc_right_left.clear()
	#patchify the image
	patches_img = patchify(imgContour, (patch_size_h, patch_size_h ,3), step=patch_size_h)

	#finding the color of fisrt left top patch in the image

	#go through all patches and find where we can see the difference between them
	for i in range(patches_img.shape[1]-1,0,-1):
		prev_patch_img = patches_img[patches_img.shape[0]-1, i, 0, :, :, :]
		prev_average_color_row = np.average(prev_patch_img, axis=0)
		prev_average_color = np.average(prev_average_color_row, axis=0)
		prev_d_img = np.ones((312,312,3), dtype=np.uint8)
		prev_d_img[:,:] = prev_average_color
		(prev_B, prev_G, prev_R)= cv2.split(prev_d_img)
		for j in range(patches_img.shape[0]-1,0,-1):
			single_patch_img = patches_img[j, i, 0, :, :, :]
			average_color_row = np.average(single_patch_img, axis=0)
			average_color = np.average(average_color_row, axis=0)
			d_img = np.ones((312,312,3), dtype=np.uint8)
			d_img[:,:] = average_color
			(B, G, R)= cv2.split(d_img)

			if not ( mse(prev_B , B) <= limit or mse(prev_G , G) <= limit or mse(prev_R , R) <= limit ):
				loc_right_left.append((j,i))
				#cv2.imwrite('colorimage_' + '_'+ str(i)+str(j)+'.jpg', d_img)
			prev_B = B
			prev_G = G
			prev_R = R

	#crop the image horizontally
	#print(loc_up_down)
	row , col = loc_right_left[0]
	cut_img = img[:  , :col*patch_size_h]
	return cut_img

def if_backgrnd_ver(img):
	check_ver = True
	patch_size =20

	patches_img = patchify(img, (patch_size, patch_size, 3), step=patch_size)

	#finding the color of fisrt left top patch in the image
	prev_patch_img = patches_img[0, patches_img.shape[1]-1, 0, :, :, :]
	prev_average_color_row = np.average(prev_patch_img, axis=0)
	prev_average_color = np.average(prev_average_color_row, axis=0)
	prev_d_img = np.ones((312,312,3), dtype=np.uint8)
	prev_d_img[:,:] = prev_average_color
	(prev_B, prev_G, prev_R)= cv2.split(prev_d_img)

	single_patch_img = patches_img[patches_img.shape[0]-1, patches_img.shape[1]-1, 0, :, :, :]
	average_color_row = np.average(single_patch_img, axis=0)
	average_color = np.average(average_color_row, axis=0)
	d_img = np.ones((312,312,3), dtype=np.uint8)
	d_img[:,:] = average_color
	(B, G, R)= cv2.split(d_img)

	if ( mse(prev_B , B) <= 5000 and mse(prev_G , G) <= 5000 and mse(prev_R , R) <= 5000):
		check_ver = False

	return check_ver


def crop_horizontally(img, imgContour, num_image, patch_size_h, limit):
	loc_up_down =[]
	#define variables
	loc_up_down.clear()
	#patchify the image
	patches_img = patchify(imgContour, (patch_size_h, patch_size_h ,3), step=patch_size_h)

	#finding the color of fisrt left top patch in the image
	prev_patch_img = patches_img[0, 0, 0, :, :, :]
	prev_average_color_row = np.average(prev_patch_img, axis=0)
	prev_average_color = np.average(prev_average_color_row, axis=0)
	prev_d_img = np.ones((312,312,3), dtype=np.uint8)
	prev_d_img[:,:] = prev_average_color
	(prev_B, prev_G, prev_R)= cv2.split(prev_d_img)

	#go through all patches and find where we can see the difference between them
	for i in range(patches_img.shape[0]):
        #finding the color of fisrt left top patch in the image
        prev_patch_img = patches_img[i, 0, 0, :, :, :]
        prev_average_color_row = np.average(prev_patch_img, axis=0)
        prev_average_color = np.average(prev_average_color_row, axis=0)
        prev_d_img = np.ones((312,312,3), dtype=np.uint8)
        prev_d_img[:,:] = prev_average_color
        (prev_B, prev_G, prev_R)= cv2.split(prev_d_img)

		for j in range(patches_img.shape[1]):
			single_patch_img = patches_img[i, j, 0, :, :, :]
			average_color_row = np.average(single_patch_img, axis=0)
			average_color = np.average(average_color_row, axis=0)
			d_img = np.ones((312,312,3), dtype=np.uint8)
			d_img[:,:] = average_color
			(B, G, R)= cv2.split(d_img)

			if not ( mse(prev_B , B) <= limit or mse(prev_G , G) <= limit or mse(prev_R , R) <= limit ):
				loc_up_down.append((i,j))
				#cv2.imwrite('colorimage_' + '_'+ str(i)+str(j)+'.jpg', d_img)
			prev_B = B
			prev_G = G
			prev_R = R

	#crop the image horizontally
	#print(loc_up_down)
	row , col = loc_up_down[0]
	cut_img = img[ (row*patch_size_h) :  , :]
	cv2.imwrite('image_'+ str(num_image)+'.jpg', cut_img)


def if_backgrnd_hor(img):
		check_hor = True
		patch_size = 20

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

		return (check_hor)


#define variables
num_image = 0

input_path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-test'
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(input_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	img = cv2.imread(os.path.join(input_path,f))
	imgContour = img.copy()
	img_v = img.copy()
	imgCrop = img.copy()

	num_image = num_image+1

	if not (if_backgrnd_ver(img)):
		imgBlur = cv2.GaussianBlur(img, (3,3), 1)
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imgCanny = auto_canny(imgGray)
		kernel = np.ones((3,3))
		imgDil = cv2.dilate(imgCanny, kernel, iterations =1)

		getContours(imgDil, imgContour)
		img_v = crop_vertically(img, imgContour, num_image, 10, 300)
	else:
		img_v = crop_vertically(img, img, num_image, 20, 350)

	if not (if_backgrnd_hor(img_v)):
		imgBlur = cv2.GaussianBlur(img_v, (3,3), 1)
		imgGray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
		imgCanny = auto_canny(imgGray)
		kernel = np.ones((3,3))
		imgDil = cv2.dilate(imgCanny, kernel, iterations =1)

		getContours(imgDil, imgContour)
		crop_horizontally(img_v, imgContour, num_image, 10, 600)
	else:
		crop_horizontally(img_v, img_v, num_image, 20, 350)



cv2.waitKey(0)
cv2.destroyAllWindows()
