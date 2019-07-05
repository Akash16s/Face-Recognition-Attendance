# Face-Recognition and attendence
Learned and implemented the Face recognition using face_recognition api

# Attendence
Using face recognition api, self implemented algo in SceneChangeDetect.py and Multithreading pool developed a system in which one can pass argument as past cctv footage and can do facial_recognition(attendence).

# Processing Speed: 
50sec(changes rapid) -> 10 sec processing and largely depends upon the visual it is getting as it skips most of the part of frame in which no change occured and also no face detected.
# Using Attendence system
1. Clone the repository into a python virtual environment with installed packages : 
       * face_recognition
       * dlib
       * imutils
       * argparse
       * opencv-python
       * pickle
 ### clone : 
 ```$ git clone https://github.com/Akash16s/Face-Recognition-Attendence.git```
 
 ### Structure of files :
 ```--BASE_DIR 
     -dataset
          - Name of person or ID
               - Photos
     -input CCTV video
  ```
  
 ### Now first create encodings.pickle file have encodings of faces 
 ``` (venv)$ python encode.py```
 
 ### Now we can perform the Attendence 
 ``` (vnev)$ python process.py -i "Location of input Video" -d "Input the date and time %Y-%m-%d %H:%M:%S"```
