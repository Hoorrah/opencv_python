import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('shapes.png')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thrashimage = cv.threshold(imgray, 240, 255, cv.THRESH_BINARY)
countours, _ = cv.findContours(thrashimage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in countours:
    #approximate a polygon and curves with specific precision
    # we use this function to approximation of the shape of a contour
    approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True),True)#curve = contour, epsilon which specifying the approximation accuracy=0.01* cv.arcLength((contour, True),True):calculate the perimeter of contour and True consider the contour to be closed
    #then we draw all contoures around the shapes
    cv.drawContours(img, [approx], 0, (0,0,0), 5)#orginal image= img,contours = [approx], contours index = 0 because we iterate on all contours, color= (0,0,0), thickness = 5
    #find coordinate of approximations
    x = approx.ravel()[0] #first argument is x coordinate
    y = approx.ravel()[1] # second argument is y coordinate
    #write the shape name
    if len(approx) == 3:
        cv.putText(img, "triangle", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0) )#src=img, text , coordinate, font, font scale, color
    elif len(approx) == 4:
        #decide is rectangle or square
        x , y , width , height = cv.boundingRect(approx)
        aspectRatio = float(width)/ height
        print (aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv.putText(img, "square", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0) )
        else:
            cv.putText(img, "rectangle", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0) )
    elif len(approx) == 5:
        cv.putText(img, "pentagon", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0) )
    elif len(approx) == 6:
        cv.putText(img, "hexagon", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0) )
    else:
        cv.putText(img, "circle", (x,y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0) )




cv.imshow('image', img)
#cv.imshow('image gray', imgray)
cv.waitKey()
cv.destroyAllWindows()
