import numpy as np
import cv2
img = cv2.imread('77_cover.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
#print( M )

x,y,w,h = cv2.boundingRect(cnt)
imgc = cv2.rectangle(img,(x,y),(x+w,y+h),(88,255,99),2)

cv2.imwrite('image.jpg',imgc)

cv2.waitKey(0)
cv2.destroyAllWindows()
