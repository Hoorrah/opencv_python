import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from patchify import patchify


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


#define variables
patch_size = 10
loc_up_down = []
#reading image
img = cv2.imread("1_cover.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# threshold, and automatically determined threshold
auto = auto_canny(blurred)
#patchify the image
patches_img = patchify(auto, (patch_size, patch_size), step=patch_size)


#go through all patches and find where we can see the difference between them
for i in range(patches_img.shape[0]):
	for j in range(patches_img.shape[1]):
		single_patch_img = patches_img[i, j, :, :, ]
		average_color_row = np.average(single_patch_img, axis=0)
		average_color = np.average(average_color_row, axis=0)
		d_img = np.ones((312,312,3), dtype=np.uint8)
		d_img[:,:] = average_color
		#loc_up_down.append((i,j))
		cv2.imwrite('colorimage_' + '_'+ str(i)+str(j)+'.jpg', d_img )

#print(loc_up_down)
cv2.waitKey(0)
cv2.destroyAllWindows()
