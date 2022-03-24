#contoures is the curve joiningall the continouse point along boundry which having same color or intensity
#good for shape analysis or object detection or object recognition

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('pic3.png')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thresholdvalue, threshimg = cv.threshold(imgray, 127, 255, 0 ) # source =img, threshold value=127, max-value-threshold = 255,thresholdtype =0
#return value of findcontoures: contoures (is pythonlist of all the contours in the image) hierarchy(parentchild relation in contoures:next,previous,fisrt,parrent)
contours, hierarchy = cv.findContours(threshimg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE )#src= threshimg, contour mode= cv.RETR_TREE, method which we want apply(contpur approximation method)=cv.CHAIN_APPROX_NONE
print("number of countours= " + str(len(contours)))


cv.drawContours(img, contours, -1, (0, 255, 0), 3)#orginal image= img,contours = contours, contours index = -1 draw all contours which found inside the image, color= (0, 255, 0), thickness = 3

cv.imshow('image', img)
cv.imshow('image gray', imgray)
cv.waitKey()
cv.destroyAllWindows()
