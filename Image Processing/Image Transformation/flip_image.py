import cv2 as cv
import numpy as np

img = cv.imread('pic2.png',1 )

#flip the image vertically
imgverflip = cv.flip(img , flipCode = 0)

#flip the image horizontally
imghorflip = cv.flip(img , flipCode = 1)

#flip the image vertically and horizontally
imgverhorflip = cv.flip(img , flipCode = -1)


#stack images in single frame
hstack1 = np.hstack((img,imghorflip)) # in the left side we see orginal image and in the right side we see flip image
hstack2 = np.hstack((imgverflip,imgverhorflip))
hstack3 = np.hstack((img,imgverhorflip))

final_image = np.vstack((hstack1,hstack2)) #for adding stacks


cv.imshow("Final Image ", final_image)
cv.waitKey(0)
cv.destroyAllWindows()
