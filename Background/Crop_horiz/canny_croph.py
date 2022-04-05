import cv2
import numpy as np
from PIL import Image
import os, os.path
import glob
from matplotlib import pyplot as plt
from patchify import patchify


def auto_canny(image, sigma=0.11):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


#calculating the Mean square error for finding the difference between colours
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /=float(imageA.shape[0] * imageA.shape[1])
	return err

#define variables
loc_up_down = []#location of patches that we can find the difference when we go through the image from up to down
patch_size = 23
num_image = 0

input_path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image-2'
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(input_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	img = cv2.imread(os.path.join(input_path,f))
	num_image = num_image + 1

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	# threshold, and automatically determined threshold
	auto = auto_canny(blurred)

	loc_up_down.clear()
	#patchify the image
	patches_img = patchify(auto, (patch_size, patch_size), step=patch_size)



	#finding the color of fisrt left top patch in the image
	prev_patch_img = patches_img[0, 0, :, :]
	prev_average_color_row = np.average(prev_patch_img, axis=0)
	prev_average_color = np.average(prev_average_color_row, axis=0)
	prev_d_img = np.ones((312,312,3), dtype=np.uint8)
	prev_d_img[:,:] = prev_average_color

	#go through all patches and find where we can see the difference between them
	for i in range(patches_img.shape[0]):
		for j in range(patches_img.shape[1]):
			single_patch_img = patches_img[i, j, :, :]
			average_color_row = np.average(single_patch_img, axis=0)
			average_color = np.average(average_color_row, axis=0)
			d_img = np.ones((312,312,3), dtype=np.uint8)
			d_img[:,:] = average_color
			#print(mse(d_img , prev_d_img))

			if not ( mse(d_img , prev_d_img) <= 3100 ):
				loc_up_down.append((i,j))
				#cv2.imwrite('colorimage_' + '_'+ str(i)+str(j)+'.jpg', d_img)
			prev_d_img = d_img


	#crop the image horizontally
	#print(loc_up_down)
	row , col = loc_up_down[0]
	cut_img = img[ (row*patch_size) :  , :]
	cv2.imwrite('cropimage_' + '_'+ str(num_image)+'.jpg', cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
