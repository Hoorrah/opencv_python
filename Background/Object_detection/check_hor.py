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


def auto_canny(image, sigma=0.33):
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


input_path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image-2'
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(input_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	img = cv2.imread(os.path.join(input_path,f))
	num_image = num_image + 1
	chck = True

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
		chck = False
	if not (chck):
		cv2.imwrite('goodimage_' + '_'+ str(num_image)+'.jpg', img)
	num_image = num_image+1

cv2.waitKey(0)
cv2.destroyAllWindows()
