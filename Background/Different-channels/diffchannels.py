import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
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


img = cv2.imread("1_cover.jpg")
#print(img.shape[0])
#print(img.shape[1])
#print(img.shape[2])
patches_img = patchify(img, (150,150,3), step=150)

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, 0, :, :, :]
        (B, G, R)= cv2.split(single_patch_img)
        Bblurred = cv2.GaussianBlur(B, (3, 3), 0)
        Gblurred = cv2.GaussianBlur(G, (3, 3), 0)
        Rblurred = cv2.GaussianBlur(R, (3, 3), 0)
        #merged = cv2.merge([B , G, R])
        Bauto = auto_canny(Bblurred)
        Gauto = auto_canny(Gblurred)
        Rauto = auto_canny(Rblurred)
        cv2.imwrite('image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img)
        cv2.imwrite('Bimage_' + '_'+ str(i)+str(j)+'.jpg', Bauto )
        cv2.imwrite('Gimage_' + '_'+ str(i)+str(j)+'.jpg', Gauto )
        cv2.imwrite('Rimage_' + '_'+ str(i)+str(j)+'.jpg', Rauto )
        #cv2.imwrite('Merged_' + '_'+ str(i)+str(j)+'.jpg', merged)
        #cv2.imshow("Patched Image",single_patch_img)


cv2.waitKey()
cv2.destroyAllWindows()
