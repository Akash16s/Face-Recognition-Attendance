import face_recognition
import cv2
import pickle
import imutils
import os
import time
from multiprocessing.pool import ThreadPool

pool1 = ThreadPool(processes = 1)
pool2 = ThreadPool(processes = 2)

def recogFace(rgb):
	matches = face_recognition.compare_faces(data["encodings"], encoding)
	return matches

def recogLoc(rgb):
	boxes = face_recognition.face_locations(rgb, model = "hog")
	encodings = face_recognition.face_encodings(rgb, boxes)
	return encodings,boxes


# Load the known face and encodings
print("[INFO] loading encodings ..")
data = pickle.loads(open("encodings.pickle","rb").read())

#inititlise the camera
cap = cv2.VideoCapture(0)
time.sleep(2.0)

#start the videocapture
while 1:
	ret, img = cap.read()
	#Convert the BGR to RGB
	# a width of 750px (to speed up processing)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(img, width = 750)
	r = img.shape[1]/float(rgb.shape[1])


	#detect boxes
	result1 = pool1.apply_async(recogLoc,(rgb,))
	encodings,boxes = result1.get()
	names = []
	
	# loop over the facial encodings
	for encoding in encodings :
		# attempt to match each face then initialise a dicationary
		#matches = face_recognition.compare_faces(data["encodings"], encoding)
		result = pool2.apply_async(recogFace,(rgb,))
		matches = result.get()
		name = "Unkown"

		# check to see if we have found a match
		if True in matches:
			#find the indexes of all matched faces then initialize a 
			# dicationary to count the total number of times each face matched
			matchedIds = [i for (i,b) in enumerate(matches) if b]
			counts ={}

			#loop over the recognized faces
			for i in matchedIds:
				name = data["names"][i]
				counts[name] = counts.get(name,0)+1

			#determine the recognized faces with largest number
			# of votes (note: in the event of an unlikely tie Python will select first entry in the dictionary)
			name = max(counts , key = counts.get)

		names.append(name)

	# loop over recognized faces
	for ((top,right,bottom,left),name)in zip(boxes, names):
		top = int(top*r)
		right = int(right*r)
		bottom = int(bottom*r)
		left = int(left *r)
		# draw the predicted face name on the image
		cv2.rectangle(img, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	cv2.imshow("Frame", img)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()