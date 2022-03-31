#adaptive thresholding is method where threshold value is calculated for smaller region so we have diffrenet threshold value for different regions

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('sudoku.png',0)
_, threshol1 = cv.threshold(img ,127, 255, cv.THRESH_BINARY)
#in ADAPTIVE_THRESH_MEAN_C the threshold value is the mean of the neighborhood area
threshol2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C , cv.THRESH_BINARY, 11, 2) #src= img, max value =255, adaptive method , thresholdtype =cv.THRESH_BINARY , blocksize (size of neighborhood area)= 11, the value of C= 2
# in ADAPTIVE_THRESH_GAUSSIAN_C the threshold is the weighted sum of neighborhood values where weights are the Guassian window
threshol3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,  cv.THRESH_BINARY, 11, 2)

titles= [ 'Image', 'Threshol1', 'Threshol2', 'Threshol3']
images = [img, threshol1, threshol2, threshol3]

for i in range(4):
    plt.subplot(2,2, i+1), plt.imshow(images[i], 'gray')#number of rows = 2, number of columns=3, index of images= i+1 ----- gray means gray scale images
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

#cv.imshow("Image",img)
#cv.imshow("Threshol1",threshol1)
#cv.imshow("Threshol2",threshol2)
#cv.imshow("Threshol3",threshol3)

#cv.waitKey(0)
#cv.destroyAllWindows()
