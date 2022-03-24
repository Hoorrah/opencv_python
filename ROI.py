#split,merge,resize,add,addweighted,ROI
#ROI: region of interest
import cv2

img = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv-logo.png')

print(img.shape) #returns tuple of number of rows columns and channels
print(img.size) #returens total number of pixels is a accessed
print(img.dtype) # returns image datatype is obtained
b, g, r = cv2.split(img)
img = cv2.merge((b,g,r))

ball = img[280:340, 330:390] #coordinate of ball
img[273:333, 100:160] = ball #copy the ball to another place in image

img = cv2.resize(img, (512,512))# resizing images
img2 = cv2.resize(img2, (512,512))


#dst = cv2.add(img2, img); #add 2 image together
dst = cv2.addWeighted(img, .1, img2, .2, 0)  #src1*alpha + src2*beta + gamma


cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


#bitwise operations
#bitAnd = cv2.bitwise_and( , )
#bitOr = cv2.bitwise_or( , )
#bitXor = cv2.bitwise_xor( , )
#bitNot = cv2.bitwise_not( )
