import cv2
import numpy as np
img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edgesimg = cv2.Canny(gray,50,150,apertureSize = 3)
#cv2.imshow('edges', edgesimg)

lines = cv2.HoughLinesP(edgesimg,1,np.pi/180,100,minLineLength=100,maxLineGap=10)#src=img, rho=1,theta=pi/180, threshold=100
for line in lines:
    x1,y1,x2,y2 = line[0]# in this method it gives us 2 coordinate so we don't need to calculate them
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2) #src=img, point1 , point 2, color ,thickness

cv2.imshow('image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
