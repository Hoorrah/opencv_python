
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path
import glob

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

#read valid_images
num_image = 0
input_path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image'
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(input_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	image = cv2.imread(os.path.join(input_path,f))
	num_image = num_image + 1
	# load the image, convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	# threshold, and automatically determined threshold
	auto = auto_canny(blurred)
	titles= [ 'Orginale image','Automatically']
	images = [image , auto]

	for i in range(2):
		plt.subplot(1,2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
		plt.savefig(str(num_image) +'.jpg')
