import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path
import glob


num_image = 0
path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image'
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img = cv.imread(os.path.join(path,f),0)
    num_image = num_image + 1

    #laplacian gradient
    lap = cv.Laplacian( img, cv.CV_64F , ksize = 5)#src = imh , datatype= cv.CV_64F (64bit float and it supports negative numbers which we are dealing with when we us laplacian method),kernel size=3
    lap_img = np.uint8(np.absolute(lap))

    #sobelx and sobely(sobel gradient representation)
    #in sobelx changing direction in intensity is in xdirection more verticl lines and in sobely changing direction in intensity is in y direction
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)#src= img, data type = cv.CV_64F ,1= xdirection sobelx , 0= ydirection
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)#src= img, data type = cv.CV_64F ,0= xdirection , 1= ydirection sobely

    sobelX_img = np.uint8(np.absolute(sobelX))
    sobelY_img = np.uint8(np.absolute(sobelY))

    #combine sobelx and sobely
    sobelcombined = cv.bitwise_or(sobelX_img, sobelY_img)

    canny = cv.Canny(img, 50, 200)#src=img, threshold value 1=100 ,threshold value 2=200 ,

    titles= [ 'Orginale image','laplacian','sovelX', 'sovelY', 'sobelcombined','canny']
    images = [img , lap_img, sobelX_img, sobelY_img, sobelcombined, canny]

    for i in range(6):
        plt.subplot(2,3, i+1), plt.imshow(images[i], 'gray')#number of rows = 2, number of columns=3, index of images= i+1 ----- gray means gray scale images
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        plt.savefig(str(num_image)+'.png')
    #cv2.imshow('image',img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
