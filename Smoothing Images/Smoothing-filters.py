#in denoising we take a kernel and kind of multiply it by progressievly moving it on your image

import cv2 as cv
import numpy as np

img = cv.imread('pic2.png',1 )

#convolution using the kernel that we have just defined
kernel = np.ones((3,3), np.float32)/9 #we creat a kernel to multiply it to image and we devide it by 5x5 to normlizing it, to not change the energy of image
filter_2D = cv.filter2D(img, -1, kernel ) #src = img , ddepth =-1 (it is desirable depth of destination image and -1 means resulting image will have same depth) , kernel =kernel
#cv2.blur is same as cv2.filter2D just give the kernel size as input
blur = cv.blur(img , (3,3))
#in cv.GaussianBlur instead of box filter it use Gaussian kernel - a little better than other 2 function
guassian_blur = cv.GaussianBlur(img, (3,3), 0)
#median in better for remain edges it is very good for sal and paper noise
median_blur = cv.medianBlur(img , 3) #src = img , kernelsize should be odd and greater than 1
#bilatral filter can reduce unwanted noise very well while keeping edges sharp
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75 ) #src=img ,diameter of each pixel =9, sigmacolor and sigmaspace =9, larger filter sigma means farther colors within pixel neighborhood will be mixed together

cv.imshow("Orginal",img)
cv.imshow("filter_2D",filter_2D)
cv.imshow("blur",blur)
cv.imshow("guassian_blur",guassian_blur)
cv.imshow("median_blur",median_blur)
cv.imshow("bilateral_blur", bilateral_blur)

cv.waitKey(0)
cv.destroyAllWindows()
