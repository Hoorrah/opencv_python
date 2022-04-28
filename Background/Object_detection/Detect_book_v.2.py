#_________ run the code in command line : python Detect_book.py (path to the data set)

import cv2
import numpy as np
from PIL import Image
import sys
import os
import glob
from matplotlib import pyplot as plt
from patchify import patchify
import argparse


#calculating the Mean square error for finding the difference between colours of patches
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /=float(imageA.shape[0] * imageA.shape[1])
	return err
#____________________________________________________________________________________________________


def auto_canny(image, sigma=0.05): #Sigma can use to vary the percentage thresholds determined based on simple statistics
	# compute the median of the single channel pixel intensities
	#the median filter considers each pixel in the image in turn and looks at its nearby neighbors to decide whether or not it is representative of its surroundings.
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v)) #define lower threshold
	upper = int(min(255, (1.0 + sigma) * v)) #define upper threshold
	edged = cv2.Canny(image, lower, upper) #calculate Canndy edge detection
	return edged# return edge detected image
#_____________________________________________________________________________________________________

#finding contours of image and drwa them on the image for more performance
def getContours(img):
	#apply Gaussian blurring to the image, which is commonly used when reducing the size of an image
	imgContour = img.copy()
	kernel = np.ones((3,3)) # define kernel szie for dilating image
	imgBlur = cv2.GaussianBlur(img, (3,3), 1) #1,source image 2.ksize: Kernal is matrix of an (no. of rows)*(no. of columns) order. sigmaX: Standard deviation value of kernal along horizontal direction
	imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY) # convert the image from color space BGR to gray space
	imgCanny = auto_canny(imgGray) #apply auto Canny edge detection function
	imgDil = cv2.dilate(imgCanny, kernel, iterations =1) # increases the object area
	#find countours of image
	contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL: only extreme outer contours , CHAIN_APPROX_NONE: we use non to get all the points
	for cnt in contours:#go through all contours
		area = cv2.contourArea(cnt)#finding area of contours
		#areaMin =cv2.getTrackbarPos("Area", "Parameters")
		if area > 1000: #if the contour area is small skip it and just draw contours that their area is more than 1000
			return cv2.drawContours(imgContour, cnt, -1, (0,0,0), 5) #1. source image, 2. the contours which should be passed as a Python list, 3. index of contours To draw all contours, pass -1. 4.color,5. thickness
#_____________________________________________________________________________________________________


#function for cheking if there is background in two side of image
def if_background( src_img , type):#get image as input_path
	#define variables
	check_result = False #if there is background we make this value True and return it
	patch_size = 20
	#type is type of we want check if there is background or not 0: horizontally , vertically right :1, vertically left: 2
	patches_img = patchify(src_img, (patch_size, patch_size, 3), step=patch_size) #image_to_patch, patch_shape, step
	#we want to compare to patches together row_first and column_fisrt are coordinate of fisrt patch and row_sec and column_sec are coordinate of second patch
	if (type == 0) : # type of crop is horizontally patch1 =top left , patch2 = top right
		row_first = 0
		row_sec = 0
		column_fisrt = 1
		column_sec = patches_img.shape[1]-1
		limit = 300 #limit for comparing mean square error for colors of patches
	elif (type == 1) : #type of crop is vertically from right partof image , patch1 =top right , patch2 = bottom right
		row_first = 0 #row of first patch
		row_sec = patches_img.shape[0]-1
		column_fisrt = patches_img.shape[1]-1
		column_sec = patches_img.shape[1]-1
		limit = 5000 #limit for comparing mean square error for colors of patches
	elif (type == 2) : #type of crop is vertically from left part of image , patch1 =top left , patch2 = bottom left
		row_first = 1
		row_sec = patches_img.shape[0]-1
		column_fisrt = 0
		column_sec = 0
		limit = 5000 #limit for comparing mean square error for colors of patches
	#finding the color of fisrt patch
	fisrt_patch_img = patches_img[row_first, column_fisrt, 0, :, :, :] # select fisrt patch
	fisrt_average_color_row = np.average(fisrt_patch_img, axis=0) # finding its color
	fisrt_average_color = np.average(fisrt_average_color_row, axis=0)
	fisrt_d_img = np.ones((312,312,3), dtype=np.uint8) #creating a matrix (window)
	fisrt_d_img[:,:] = fisrt_average_color #copy the color to matrix
	(fisrt_B, first_G, first_R)= cv2.split(fisrt_d_img) # split the window to its color channels
	#finding the color of second patch
	sec_patch_img = patches_img[row_sec, column_sec, 0, :, :, :] #select second patch
	sec_average_color_row = np.average(sec_patch_img, axis=0) # finding its color
	sec_average_color = np.average(sec_average_color_row, axis=0)
	sec_d_img = np.ones((312,312,3), dtype=np.uint8) #creating a matrix (window)
	sec_d_img[:,:] = sec_average_color #copy the color to matrix
	(sec_B , sec_G, sec_R)= cv2.split(sec_d_img) # split the window to its color channels
	#calculating the MSE between color of tow patches
	if ( mse(fisrt_B , sec_B) <= limit and mse(first_G , sec_G) <= limit and mse(first_R , sec_R) <= limit):
		check_result = True #there is background
	return check_result
#_____________________________________________________________________________________________________


#function for crpping image background vertically from right of image to left part
def crop_vertically_RtoL(img, imgContour, num_image, patch_size_v, limit): #1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles
	#define variables
	loc_right_left =[] #list of where the patches are different
	loc_right_left.clear()
	#patchify the orginal image
	patches_img = patchify(imgContour, (patch_size_v, patch_size_v ,3), step=patch_size_v)
	#go through all patches and find where we can find location of patches that there is the difference between them (new area)
	for j in range(patches_img.shape[1]-1,0,-1): #column by column from right of image to left (j)
		prev_patch_img = patches_img[patches_img.shape[0]-1, j, 0, :, :, :] #last row and jth column
		prev_average_color_row = np.average(prev_patch_img, axis=0) # finding its color
		prev_average_color = np.average(prev_average_color_row, axis=0)
		prev_d_img = np.ones((312,312,3), dtype=np.uint8) #creating a matrix (window)
		prev_d_img[:,:] = prev_average_color #copy the color to matrix
		(prev_B, prev_G, prev_R)= cv2.split(prev_d_img) # split the window to its color channels
		for i in range(patches_img.shape[0]-1,0,-1): #row by row from bottom to top
			single_patch_img = patches_img[i, j, 0, :, :, :] #ith row and jth column
			average_color_row = np.average(single_patch_img, axis=0)# finding its color
			average_color = np.average(average_color_row, axis=0)
			d_img = np.ones((312,312,3), dtype=np.uint8)#creating a matrix (window)
			d_img[:,:] = average_color #copy the color to matrix
			(B, G, R)= cv2.split(d_img)# split the window to its color channels
			#checking the MSE of color of each patch with its previous of it
			if not ( mse(prev_B , B) <= limit or mse(prev_G , G) <= limit or mse(prev_R , R) <= limit ):
				loc_right_left.append((i,j)) # add the location of the patch to the list
			prev_B = B
			prev_G = G
			prev_R = R
	row , col = loc_right_left[0] #location of fisrt patches that they are different
	return img[:  , :col*patch_size_v]  #return cropped image
#_____________________________________________________________________________________________________


#function for crpping image background vertically from left of image to right part
def crop_vertically_LtoR(img, imgContour, num_image, patch_size_v, limit): #1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles
	#define variables
	loc_left_right =[] #list of where the patches are different
	loc_left_right.clear()
	#patchify the orginal image
	patches_img = patchify(imgContour, (patch_size_v, patch_size_v ,3), step=patch_size_v)
	#go through all patches and find where we can find location of patches that there is the difference between them (new area)
	for j in range(patches_img.shape[1]-1): #column by column from right of image to left (j)
		prev_patch_img = patches_img[patches_img.shape[0]-1, j, 0, :, :, :] #last row and jth column
		prev_average_color_row = np.average(prev_patch_img, axis=0) # finding its color
		prev_average_color = np.average(prev_average_color_row, axis=0)
		prev_d_img = np.ones((312,312,3), dtype=np.uint8) #creating a matrix (window)
		prev_d_img[:,:] = prev_average_color #copy the color to matrix
		(prev_B, prev_G, prev_R)= cv2.split(prev_d_img) # split the window to its color channels
		for i in range(patches_img.shape[0]-1,0,-1): #row by row from bottom to top
			single_patch_img = patches_img[i, j, 0, :, :, :] #ith row and jth column
			average_color_row = np.average(single_patch_img, axis=0)# finding its color
			average_color = np.average(average_color_row, axis=0)
			d_img = np.ones((312,312,3), dtype=np.uint8)#creating a matrix (window)
			d_img[:,:] = average_color #copy the color to matrix
			(B, G, R)= cv2.split(d_img)# split the window to its color channels
			#checking the MSE of color of each patch with its previous of it
			if not ( mse(prev_B , B) <= limit or mse(prev_G , G) <= limit or mse(prev_R , R) <= limit ):
				loc_left_right.append((i,j)) # add the location of the patch to the list
			prev_B = B
			prev_G = G
			prev_R = R
	row , col = loc_left_right[0] #location of fisrt patches that they are different
	return img[:  , col*patch_size_v:]  #return cropped image
#_____________________________________________________________________________________________________



#function for crpping image background horizontally
def crop_horizontally(img, imgContour, num_image, patch_size_h, limit):#1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles
	#define variables
	loc_up_down =[] #list of where the patches are different
	loc_up_down.clear()
	#patchify the orginal image
	patches_img = patchify(imgContour, (patch_size_h, patch_size_h ,3), step=patch_size_h)
	#finding the color of fisrt left top patch in the image
	prev_patch_img = patches_img[0, 0, 0, :, :, :] # top left patch
	prev_average_color_row = np.average(prev_patch_img, axis=0)# finding its color
	prev_average_color = np.average(prev_average_color_row, axis=0)
	prev_d_img = np.ones((312,312,3), dtype=np.uint8)#creating a matrix (window)
	prev_d_img[:,:] = prev_average_color #copy the color to matrix
	(prev_B, prev_G, prev_R)= cv2.split(prev_d_img) # split the window to its color channels
	#go through all patches and find where we can see the difference between them
	for i in range(patches_img.shape[0]):
		for j in range(patches_img.shape[1]):
			single_patch_img = patches_img[i, j, 0, :, :, :]
			average_color_row = np.average(single_patch_img, axis=0) # finding its color
			average_color = np.average(average_color_row, axis=0)
			d_img = np.ones((312,312,3), dtype=np.uint8)#creating a matrix (window)
			d_img[:,:] = average_color #copy the color to matrix
			(B, G, R)= cv2.split(d_img) # split the window to its color channels
			#checking the MSE of color of each patch with its previous of it
			if not ( mse(prev_B , B) <= limit or mse(prev_G , G) <= limit or mse(prev_R , R) <= limit ):
				loc_up_down.append((i,j))  # add the location of the patch to the list
			prev_B = B
			prev_G = G
			prev_R = R
	row , col = loc_up_down[0] #location of fisrt patches that they are different
	return img[ (row*patch_size_h) :  , :] #crop the image horizontally



#########################################################################################################
#define variables
num_image = 0 # number of images
#reading images from folder
#getting path from commandline
#input_path = sys.argv[1]
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to folder of input images")
args = vars(ap.parse_args())
input_path = args["image"]
# Check if path exits
if os.path.exists(input_path):
    print("Folder exists")
else :
	print("Folder does not exist")

valid_images = [".jpg",".gif",".png",".tga"] #define valid image type
for f in os.listdir(input_path):# go through folder path and read images one by one
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images: # chcek if the image type is valid or not
		print (" Error : Invalid Image type _ This file is not processed : " + str(os.path.join(input_path,f)))
		continue
	#catch error if it couldn't read file
	try :
		img = cv2.imread(os.path.join(input_path,f))# reading the image from the path
	except cv2.error as e:
		print ("The image cannot be read" + str(os.path.join(input_path,f)))#print error
		continue
	#increasing number of image
	num_image = num_image+1
	#checking if the image has background in the right part of it
	if (if_background(img , type = 1)): #1.source image 2. type of cropping is vertically from right of image
		imgContour = getContours(img) #finding contours of image and draw them on the imgContour
		img_v = crop_vertically_RtoL(img, imgContour, num_image, patch_size_v=10, limit=100 )#1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles , crop from right to left
	else: #there is no background
		img_v = crop_vertically_RtoL(img, img, num_image, patch_size_v=10, limit=100)#1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles crop from right to left
	if (if_background(img , type = 2)): #1.source image 2. type of cropping is vertically from right of image
		imgContour = getContours(img_v) #finding contours of image and draw them on the imgContour
		img_v = crop_vertically_LtoR(img_v, imgContour, num_image, patch_size_v=10, limit=100 )#1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles  crop from left to right
	#checking if the image has background in the top part of it
	if (if_background(img_v , type = 0)): #1.source image which is croppedvertically befor 2. type of cropping is vertically from right of image
		imgContour = getContours(img_v) #finding contours of image and draw them on the imgContour
		imgCrop = crop_horizontally(img_v, imgContour, num_image, patch_size_h=10, limit=100)#1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles
		cv2.imwrite('image_'+ str(num_image)+'.jpg', imgCrop) # write the result in with jpg fromat
	else: #there is no background
		imgCrop = crop_horizontally(img_v, img_v, num_image, patch_size_h=20, limit=100)#1.orginal image , 2.image with its contours 3.number of image 4.patch szie 5.limit for mean squar error of color channeles
		cv2.imwrite('image_'+ str(num_image)+'.jpg', imgCrop) # write the result in with jpg fromat

cv2.waitKey(0)
cv2.destroyAllWindows()
