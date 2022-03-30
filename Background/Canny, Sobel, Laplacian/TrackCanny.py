import cv2

cv2.namedWindow('window')

def nothing(x):
	pass

cv2.createTrackbar('lower', 'window', 0, 255, nothing)
cv2.createTrackbar('upper', 'window', 0, 255, nothing)

path = r'C:\Users\Windows10\Desktop\Internship\Image\Book-image\1_cover.jpg'
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:

	#img = cv2.blur(img, (3,3))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	x = cv2.getTrackbarPos('lower', 'window')
	y = cv2.getTrackbarPos('upper', 'window')

	edge = cv2.Canny(gray, x, y)

	cv2.imshow('window', edge)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		break
