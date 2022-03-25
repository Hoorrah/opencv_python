#Canny edge detection algorithm is composed of 5 steps
#1.noise reduction 2.gradient calculation 3.non-maximum suppression 4.double threshold 5.edge tracking by hysteresis

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg',0)

canny = cv.Canny(img, 100, 200)#src=img, threshold value 1=100 ,threshold value 2=200 ,

titles= [ 'Orginale image','canny']
images = [img, canny ]

for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')#number of rows = 2, number of columns=3, index of images= i+1 ----- gray means gray scale images
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
