#it is technique to detect any shape if you can represent that shape in a mathematical form
#even if it is broken or distorted
#line can peresnt in two way : cartesian coordinate system (y=mx+c) mc space but not abale to show vertical lines _  Polar coordinate system (xcos+ysin=r) r theta space
#steps : 1.edge detection 2.mapping of edge points to hough space
#3.interpretation of accumulator of yield lines of infinite arcLength 4. conversion of infinite to finite lines
import cv2
import numpy as np

img = cv2.imread('sudoku.png')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # in edge detection we prefer grayscale images
edgesimage = cv2.Canny(grayimg, 50, 150, apertureSize=3) #src=grayimage ,fisrt threshold = 50, second threshold = 150,
cv2.imshow('edges', edgesimage)
lines = cv2.HoughLines(edgesimage, 1, np.pi / 180, 200) # src=edgesiage ,rho = 1,theta =np.pi / 180, threshold=200 just lines that has threshold greater than this will return

for line in lines:
    #rho :distance from (0,0)that is top left corner of image _theta: line rotation angle in radians
    rho,theta = line[0]
    x0 = np.cos(theta) * rho
    y0 = np.sin(theta) * rho
    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    x1 = int(x0 + 1000 * (-np.sin(theta)))
    # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
    y1 = int(y0 + 1000 * (np.cos(theta)))
    # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
    x2 = int(x0 - 1000 * (-np.sin(theta)))
    # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
    y2 = int(y0 - 1000 * (np.cos(theta)))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2) #src= img, we need 2 point to draw line , color and thickness


cv2.imshow('image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
