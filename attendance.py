import face_recognition
import cv2
import pickle
import imutils
from os import path
from datetime import datetime
import time
from multiprocessing.pool import ThreadPool
from SceneChangeDetect import sceneChangeDetect
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",required = True, help = "path to input video")
ap.add_argument("-e", "--encoding", help = "Path to the encodings.pickle file",default = "encodings.pickle")
ap.add_argument("-f", "--frame", help = "Frame Rate of the video", default = 30, type = int)
ap.add_argument("-d", "--date", help="Input the date and time %Y-%m-%d %H:%M:%S", type=str, required=True)
args = vars(ap.parse_args())

pool1 = ThreadPool(processes = 1)
pool2 = ThreadPool(processes = 2)
pool3 = ThreadPool(processes = 3)

# It returns the matchedIds of the face in the frame using known encodings
def recogFace(data,encoding):
	return face_recognition.compare_faces(data["encodings"], encoding)

# It returns the encodings values of the face in the frame 
def recogEncodings(rgb,boxes):
	return face_recognition.face_encodings(rgb, boxes)

# It returns the boxes of the face locations in the frame
def recogLoc(rgb):
	return face_recognition.face_locations(rgb, model = "hog")

# It returns the seconds of the Person found in the video
def getTimeOfEntry(timeOfVideo,frameNo):
	return timeOfVideo + int(frameNo/args["frame"])

if __name__ == "__main__":
	# Load the known face and encodings
	print("[INFO] loading encodings ..")
	data = pickle.loads(open(args["encoding"],"rb").read())

	#inititlise the camera
	print("[INFO] processing video...")
	# It have the utc sec of the video Creation timestamp
	dt = datetime.strptime(args["date"], '%Y-%m-%d %H:%M:%S') 
	videoCreationTime = time.mktime(dt.timetuple()) + int(19800)
	# printing the start time of the video in CCTV session
	print(datetime.utcfromtimestamp(videoCreationTime).strftime('%Y-%m-%d %H:%M:%S'))
	
	cap = cv2.VideoCapture(args["input"])
	time.sleep(1.0)

	# For calculating the time of the execution
	start_time = time.time()

	
	Attendees_Names = {}
	Unkown_attendees ={}
	frame = 0 # to count frame
	Scene = sceneChangeDetect() 

	while 1:
		encodings = [] # Strores the encodings and initialised with zero everytime
		frame +=1
		
		(grabbed, img) = cap.read()

		if not grabbed:
			break

		if(Scene.detectChange(img) == True):  # To detect some change in the frames
			#Convert the BGR to RGB
			# a width of 750px (to speed up processing)
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			rgb = imutils.resize(img, width = 750)

			boxes = [] 
			#detect boxes
			if(frame%10==0):  # Process on every nth frame
				boxes = pool1.apply_async(recogLoc,(rgb,)).get()
				if(boxes!=[]):   #If person location found then process
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
					if(name=="Unkown"):
						dataOfPresence = getTimeOfEntry(videoCreationTime, frame)
						dateAndTime={}
						dateAndTime[str(datetime.utcfromtimestamp(dataOfPresence).strftime("%Y-%m-%d"))] = datetime.utcfromtimestamp(dataOfPresence).strftime('%H:%M:%S')
						UnkownName = "Unkown"+str(frameNo)
						Unkown_attendees[UnkownName] = dateAndTime

					elif(name not in Attendees_Names):
						dataOfPresence = getTimeOfEntry(videoCreationTime, frame)
						dateAndTime={}
						dateAndTime[str(datetime.utcfromtimestamp(dataOfPresence).strftime("%Y-%m-%d"))] = datetime.utcfromtimestamp(dataOfPresence).strftime('%H:%M:%S')
						Attendees_Names[name]=dateAndTime

	cap.release()
	print("Known Attendees_Names :"+ str(Attendees_Names))
	print("Unkown Attendees_Names :" + str(Unkown_attendees))
	print("Execution Time :--- %s seconds ---" % (time.time() - start_time))


