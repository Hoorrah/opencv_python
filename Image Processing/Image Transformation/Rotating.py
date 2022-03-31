import cv2 as cv
import numpy as np

img = cv.imread('pic2.png',1 )

#rotate image by angle
imgrotate = cv.rotate(img, rotateCode = 0 ) #src =img , rotate code 0 = 90 clockwise degree 1=180 degree and 2 = 270 degree

#image center/shape
height , widtht, channel = img.shape
center = (height//2 , weight//2)
#find rotation matrix and final image
rotation_matrix = cv.getRotationMatrix2D(center, 15, 0.75) #center =center, angle= 15 degree (degree can also be negative), scalling parameter = 1.0 it is for cropping area
final_rotated = cv.warpAffine(img,rotation_matrix, (weight,height) ) #src=imh, rotation matrix = rotation_matrix, warp_dst.size = (weight, height) which describe size of output

cv.imshow("Orginal", img)
cv.imshow("imgrotate", imgrotate)
cv.imshow("final_rotated", final_rotated)
cv.waitKey(0)
cv.destroyAllWindows()
