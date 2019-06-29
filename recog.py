import face_recognition
import cv2
import pickle
import imutils
import os
import datetime
import time
from multiprocessing.pool import ThreadPool

pool1 = ThreadPool(processes = 1)
pool2 = ThreadPool(processes = 2)
pool3 = ThreadPool(processes = 3)

def recogFace(data,encoding):
	return face_recognition.compare_faces(data["encodings"], encoding)

def recogEncodings(rgb,boxes):
	return face_recognition.face_encodings(rgb, boxes)

def recogLoc(rgb):
	return face_recognition.face_locations(rgb, model = "hog")


# Load the known face and encodings
print("[INFO] loading encodings ..")
data = pickle.loads(open("encodings.pickle","rb").read())

#inititlise the camera
cap = cv2.VideoCapture(0)
time.sleep(2.0)

encodings = []
boxes = []
Attendees_Names = {}
frame = 0
#start the videocapture
while 1:
	frame +=1
	if(frame==100):
		frame = 0

	ret, img = cap.read()
	#Convert the BGR to RGB
	# a width of 750px (to speed up processing)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(img, width = 750)
	r = img.shape[1]/float(rgb.shape[1])

	#detect boxes
	if(frame%5 == 0):
		boxes = pool1.apply_async(recogLoc,(rgb,)).get()
		encodings = pool3.apply_async(recogEncodings,(rgb,boxes,)).get()
	names = []
	
	# loop over the facial encodings
	for encoding in encodings :
		# attempt to match each face then initialise a dicationary
		#matches = face_recognition.compare_faces(data["encodings"], encoding)
		matches = pool2.apply_async(recogFace,(data,encoding,)).get()
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
			if(name not in Attendees_Names):
				dataOfPresence = {"Present":str(datetime.datetime.now())}
				Attendees_Names[name]=dataOfPresence

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

print(Attendees_Names)
cv2.destroyAllWindows()
