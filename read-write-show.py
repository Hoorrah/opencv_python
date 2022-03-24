#read and write and show images

import cv2

img = cv2.imread('lena.jpg', 0 ) # read image and second element is flag (0:gary , 1:color , -1: no change)
print(img) # image matrix

cv2.imshow('image' , img ) #show image in milisec
k = cv2.waitKey(0) #wait for 5sec - and if give 0 input will not close the window until manually close it

if k == 27:#press scape
    cv2.destroyAllWindows()
elif k == ord('s'):#somebody press s save it
    cv2.imwrite('lena_copy.png' , img)#creat new image by name first input in the file
    cv2.destroyAllWindows()
#cv2.destroyWindow()
