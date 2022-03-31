##morphological transformations are some simple operations based on the image shape in grayscale
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('smarties.png',0)

_, mask = cv.threshold(img ,220 , 255, cv.THRESH_BINARY_INV) # source =img, threshold value=220, max-value-threshold = 255,thresholdtype =cv.THRESH_BINARY
#define kernel
kernel = np.ones((5,5), np.uint8)
#this method consist of convolving an image A with some kernel
dilation = cv.dilate(mask, kernel, iterations = 2)#image=mask , kernel, number of iterations by defaulte is one
# in this methos the pixel in orginal image will consider as 1 if only all pixels unedr kernel is 1
erosion = cv.erode(mask, kernel ,iterations=2)#image=mask , kernel, number of iterations by defaulte is one
#opening is another name of erosionfollowd by dilations : erosion + dilation
opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)#image=mask , type of morhologucal operation= cv.MORPH_OPEN, kernel
# dilation +erosion
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

titles= ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing']
images = [img, mask, dilation ,erosion, opening, closing]

for i in range(6):
    plt.subplot(2 , 3 , i+1), plt.imshow(images[i], 'gray')#number of rows = 2, number of columns=3, index of images= i+1 ----- gray means gray scale images
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
