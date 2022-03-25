#in matplotlib we can see x and y coordinate of image by changing mouse point and it is not fix window

import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('lena.jpg', -1)
cv.imshow('Orginal', img)

#to show image by matplotlib
# notice opencv read image in BGR format but plt read in RBG format so we should convert it
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
#for hide the tick value on X and Y axis
plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
