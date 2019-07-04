import face_recognition
import cv2
import pickle
import imutils
import os
import datetime
import time
from multiprocessing.pool import ThreadPool
from SceneChangeDetect import sceneChangeDetect
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",required = True, help = "path to input video")
ap.add_argument("-e", "--encoding", help = "Path to the encodings.pickle file",default = "encodings.pickle")
args = vars(ap.parse_args())

pool1 = ThreadPool(processes = 1)
pool2 = ThreadPool(processes = 2)
pool3 = ThreadPool(processes = 3)

def recogFace(data,encoding):
	return face_recognition.compare_faces(data["encodings"], encoding)

def recogEncodings(rgb,boxes):
	return face_recognition.face_encodings(rgb, boxes)

def recogLoc(rgb):
	return face_recognition.face_locations(rgb, model = "hog")

if __name__ == "__main__":
	# Load the known face and encodings
	print("[INFO] loading encodings ..")
	data = pickle.loads(open(args["encoding"],"rb").read())

	#inititlise the camera
	print("[INFO] processing video...")
	cap = cv2.VideoCapture(args["input"])
	time.sleep(1.0)

	start_time = time.time()

	encodings = []
	Attendees_Names = {}
	frame = 0
	Scene = sceneChangeDetect()

	while 1:
		frame +=1
		print(frame)
		
		(grabbed, img) = cap.read(5)

		if not grabbed:
			break

		if(Scene.detectChange(img) == True):
			#Convert the BGR to RGB
			# a width of 750px (to speed up processing)
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			rgb = imutils.resize(img, width = 750)

			boxes = []
			#detect boxes
			if(frame%10==0):
				boxes = pool1.apply_async(recogLoc,(rgb,)).get()
				if(boxes!=[]):
					encodings = pool3.apply_async(recogEncodings,(rgb,boxes,)).get()
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

	cap.release()
	print(Attendees_Names)
	print("--- %s seconds ---" % (time.time() - start_time))


