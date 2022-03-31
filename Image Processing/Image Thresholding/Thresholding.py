#thresholding is the binarization of an image , we select and value and before that value set everything to 0 and more than that value set them to 255

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('gradient.png',0)
# in THRESH_BINARY values set to 0 or 255
_, threshol1 = cv.threshold(img ,50, 255, cv.THRESH_BINARY) # source =img, threshold value=127, max-value-threshold = 255,thresholdtype =cv.THRESH_BINARY
# in THRESH_BINARY_INV get inverse resault of THRESH_BINARY
_, threshol2 = cv.threshold(img ,200, 255, cv.THRESH_BINARY_INV)
# in cv.THRESH_TRUNC up to threshold values of pixels will not change and afetr that the pixels value will remain same as threshold value
_, threshol3 = cv.threshold(img ,127, 255, cv.THRESH_TRUNC)
# in THRESH_TOZERO up to threshold values of pixels will be 0 and after threshold they will not change
_, threshol4 = cv.threshold(img ,127, 255, cv.THRESH_TOZERO)
_, threshol5 = cv.threshold(img ,127, 255, cv.THRESH_TOZERO_INV)


titles= [ 'Orginale image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, threshol1, threshol2, threshol3, threshol4, threshol5 ]

for i in range(6):
    plt.subplot(2,3, i+1), plt.imshow(images[i], 'gray')#number of rows = 2, number of columns=3, index of images= i+1 ----- gray means gray scale images
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

#cv.imshow("Image",img)
#cv.imshow("Threshol1",threshol1)
#cv.imshow("Threshol2",threshol2)
#cv.imshow("Threshol3",threshol3)
#cv.imshow("Threshol4",threshol4)
#cv.imshow("Threshol5",threshol5)

#cv.waitKey(0)
#cv.destroyAllWindows()
