import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from patchify import patchify

img = cv2.imread("30_cover.jpg")
#print(img.shape[0])
#print(img.shape[1])
#print(img.shape[2])
patches_img = patchify(img, (10,10,3), step=10)

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, 0, :, :, :]
        cv2.imwrite('image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img)
        #cv2.imshow("Patched Image",single_patch_img)


cv2.waitKey()
cv2.destroyAllWindows()
