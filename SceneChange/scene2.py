import cv2
import numpy as np
import math

cap =cv2.VideoCapture(0)

# To count changes
i=0
# Previous Frame mean
mean_previous = 0
# Total sum of the previous frames
sum_frame = 0

while 1:
	#Capturing the video
	ret, img = cap.read()
	# Converting to Gray Scale image
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Applying the gaussian filter on the image to reduce the noise
	blur = cv2.GaussianBlur(gray,(5,5),0)
	# Applying the edge detection technique called Canny
	edges = cv2.Canny(blur,100,150,apertureSize = 3)

	# calculating the mean of the image
	mean_now = edges.mean()

	# The sum of all previous mean
	sum_frame = (sum_frame+mean_now)/2 + 4

	# Applying this fomula to increase the substraction distance from previous frame
	sub = math.sqrt(abs(mean_now**4 - mean_previous**4) +100)
	#print(sub)
	# We now evaluate the sub with the previous all mean frames 
	if(sub > sum_frame):
		# Print if there is a change
		print("There is a change :" + str(i))
		i=i+1
	
	#print("sum_frame is "+str(sum_frame)+"  sub is "+str(sub))
	
	mean_previous=mean_now

	cv2.imshow("frame", edges)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
		
cv2.destroyAllWindows()