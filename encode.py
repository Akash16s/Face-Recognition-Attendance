from imutils import paths
import face_recognition
import pickle
import cv2 
import os

# Grabbing the input images in our dataset 
print("[INFO] quantifying faces..")
imagePaths = list(paths.list_images("dataset"))

# initialise the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	#Extract person name from the image path
	print("[INFO] processing image {}/{}".format(i+1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image and convert it frim RGM (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#detect the coordinates of face
	box = face_recognition.face_locations(rgb, model = 'hog')

	# Now as we have face coordinates now compute the encodings
	encodings = face_recognition.face_encodings(rgb, box)

	# Loop over the encodings :
	for encoding in encodings:
		# add the values and names in the empty list
		knownEncodings.append(encoding)
		knownNames.append(name)
# now save the list
print("[INFO] Serializing encodings..")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle","wb")
f.write(pickle.dumps(data))
f.close()
