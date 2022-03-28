#image gradient is a directional change in the intensity or color in an images

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image\5515_cover.jpg'

img = cv.imread(path,0)

#laplacian gradient
lap = cv.Laplacian( img, cv.CV_64F , ksize = 3)#src = imh , datatype= cv.CV_64F (64bit float and it supports negative numbers which we are dealing with when we us laplacian method),kernel size=3
lap_img = np.uint8(np.absolute(lap))

#sobelx and sobely(sobel gradient representation)
#in sobelx changing direction in intensity is in xdirection more verticl lines and in sobely changing direction in intensity is in y direction
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)#src= img, data type = cv.CV_64F ,1= xdirection sobelx , 0= ydirection
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)#src= img, data type = cv.CV_64F ,0= xdirection , 1= ydirection sobely

sobelX_img = np.uint8(np.absolute(sobelX))
sobelY_img = np.uint8(np.absolute(sobelY))

#combine sobelx and sobely
sobelcombined = cv.bitwise_or(sobelX_img, sobelY_img)

sharp_img = cv.bgsegm.createBackgroundSubtractorMOG().apply(img)

titles= [ 'Orginale image','laplacian','sovelX', 'sovelY', 'sobelcombined','sharp']
images = [img , lap_img, sobelX_img, sobelY_img, sobelcombined, sharp_img]

for i in range(6):
    plt.subplot(2,3, i+1), plt.imshow(images[i], 'gray')#number of rows = 2, number of columns=3, index of images= i+1 ----- gray means gray scale images
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
