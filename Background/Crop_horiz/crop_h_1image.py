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

#define variables
loc_up_down = []#location of patches that we can find the difference when we go through the image from up to down
patch_size = 25

#reading image
img = cv2.imread("30_cover.jpg")

#patchify the image
patches_img = patchify(img, (patch_size, patch_size ,3), step=patch_size)

#finding the color of fisrt left top patch in the image
prev_patch_img = patches_img[0, 0, 0, :, :, :]
prev_average_color_row = np.average(prev_patch_img, axis=0)
prev_average_color = np.average(prev_average_color_row, axis=0)
prev_d_img = np.ones((312,312,3), dtype=np.uint8)
prev_d_img[:,:] = prev_average_color
(prev_B, prev_G, prev_R)= cv2.split(prev_d_img)

#go through all patches and find where we can see the difference between them
for i in range(patches_img.shape[0]):
	for j in range(patches_img.shape[1]):
		single_patch_img = patches_img[i, j, 0, :, :, :]
		average_color_row = np.average(single_patch_img, axis=0)
		average_color = np.average(average_color_row, axis=0)
		d_img = np.ones((312,312,3), dtype=np.uint8)
		d_img[:,:] = average_color
		(B, G, R)= cv2.split(d_img)

		if not ( mse(prev_B , B) <= 1000 or mse(prev_G , G) <= 1000 or mse(prev_R , R) <= 1000 ):
			loc_up_down.append((i,j))
			#cv2.imwrite('colorimage_' + '_'+ str(i)+str(j)+'.jpg', d_img)
		prev_B = B
		prev_G = G
		prev_R = R

#crop the image horizontally
#print(loc_up_down)
row , col = loc_up_down[0]
cut_img = img[ (row*patch_size) :  , :]


cv2.imshow("cut image", cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
