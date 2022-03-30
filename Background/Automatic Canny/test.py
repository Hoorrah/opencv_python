
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path
import glob

def auto_canny(image, sigma=0.55):
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

image = cv2.imread(r'C:\Users\Windows10\Desktop\Internship\Image\Book-image\1031896_cover.jpg')
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
	plt.savefig( '0.55.jpg')
