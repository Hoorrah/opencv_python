import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from patchify import patchify

img = cv2.imread("1_cover.jpg")
#print(img.shape[0])
#print(img.shape[1])
#print(img.shape[2])
patches_img = patchify(img, (50,50,3), step=50)
print(patches_img.shape[0])
print(patches_img.shape[1])
#print(iz.shape[1])
num = 1
for i in range(patches_img.shape[1]-1,0,-1):
	for j in range(patches_img.shape[0]-1,0,-1):
		single_patch_img = patches_img[j, i, 0, :, :, :]
		cv2.imwrite('image_' + str(num)+'.jpg', single_patch_img)
		num = num +1
		#cv2.imshow("Patched Image",single_patch_img)

cv2.imshow("118",patches_img[11, 8, 0, :, :, :])
cv2.waitKey()
cv2.destroyAllWindows()
