#draw geometric shapes on images

import cv2
import numpy as np
img = cv2.imread('lena.jpg', 1 )
#img = np.zeros([512,512,512] , np.uint8 )    #creat black image by numpy: size, datatype


img = cv2.line(img, (0,0), (255,255), (0,255,0), 5 )  #1:image,2:pt1,3:pt3,4:color blue green red bgr format, 5:thickness 1 to
img = cv2.arrowedLine(img, (0,255 ), (255,255), (147,96,44), 5 )

img = cv2.rectangle(img, (384,0), (510,128), (0,0,255), 10 ) #point 1 is top left ,point 2 is lower right , and if for thickness give -1 it will fill rectngle
img = cv2.circle(img ,(477,63), 63, (0,255,0), -1) #center and radius

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'Hello', (10, 500) , font, 4, (255,255,255) , cv2.LINE_AA) #where to write and font face and font size and color and thickness and line type


cv2.imshow('image' , img )

k = cv2.waitKey(0)
cv2.destroyAllWindows()
