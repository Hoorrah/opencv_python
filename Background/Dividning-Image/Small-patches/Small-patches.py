import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, os.path
import glob

#reading image
img =  cv2.imread("1_cover.jpg")
image_copy = img.copy()
#finding shape of image
imgheight =img.shape[0]
imgwidth =img.shape[1]


patches_height = 30
patches_width = 30
x1 = 0
y1 = 0

for y in range(0, imgheight, patches_height ):#start value=0 , stop value= imgheight , stepsize= patches-height
    for x in range(0, imgwidth, patches_width):#start value=0 , stop value= imgheight , stepsize= patches-width
        #checking if go through all the image
        if (imgheight - y) < patches_height or (imgwidth - x) < patches_width:
            break
        y1 = y + patches_height
        x1 = x + patches_width

        # check whether the patch width or height exceeds the image width or height
        if x1 >= imgwidth and y1 >= imgheight:
            x1 = imgwidth - 1
            y1 = imgheight - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+patches_height , x:x+patches_width]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif y1 >= imgheight: # when patch height exceeds the image height
            y1 = imgheight - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+patches_height , x:x+patches_width]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif x1 >= imgwidth: # when patch width exceeds the image width
            x1 = imgwidth - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+patches_height , x:x+patches_width]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        else:
            #Crop into patches of size MxN
            tiles = image_copy[y:y+patches_height , x:x+patches_width]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

#Save full image into file directory
cv2.imshow("Patched Image",img)
cv2.imwrite("30.jpg",img)

cv2.waitKey()
cv2.destroyAllWindows()
