import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from patchify import patchify

#define variables
patch_size = 10
loc_up_down = []
#reading image
img = cv2.imread("30_cover.jpg")

#patchify the image
patches_img = patchify(img, (patch_size, patch_size ,3), step=patch_size)


#go through all patches and find where we can see the difference between them
for i in range(patches_img.shape[0]):
	for j in range(patches_img.shape[1]):
		single_patch_img = patches_img[i, j, 0, :, :, :]
		average_color_row = np.average(single_patch_img, axis=0)
		average_color = np.average(average_color_row, axis=0)
		d_img = np.ones((312,312,3), dtype=np.uint8)
		d_img[:,:] = average_color
		loc_up_down.append((i,j))
		#cv2.imwrite('colorimage_' + '_'+ str(i)+str(j)+'.jpg', d_img)

print(loc_up_down)
cv2.waitKey(0)
cv2.destroyAllWindows()
