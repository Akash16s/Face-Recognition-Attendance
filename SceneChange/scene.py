import cv2
import numpy as np
cap = cv2.VideoCapture(0)

small_sub =1000
mean_prev = 0
while 1:
	d, img = cap.read()
	img_mean = img.mean()
	#print(img_mean-mean_prev)
	sub = img_mean - mean_prev
	if(small_sub>sub):
		small_sub=sub
	cv2.imshow("Frame", img)
	mean_prev = img_mean
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
cv2.destroyAllWindows()
print((small_sub))