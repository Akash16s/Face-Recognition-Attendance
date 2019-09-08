# Face-Recognition and Attendance
Learned and implemented the Face detection and recognition using face_recognition api

# Attendance
Using face recognition api, self implemented algo in SceneChangeDetect.py and Multithreading pool developed a system in which one can pass argument as past cctv footage and can do facial_recognition(attendance).

# Processing Speed: 
50sec(changes rapid) -> 10 sec processing and largely depends upon the visual it is getting as it skips most of the part of frame in which no change occured and also no face detected.

# Using Attendance system
1. Clone the repository into a python virtual environment with installed packages : 
       * face_recognition
       * dlib
       * imutils
       * argparse
       * opencv-python
       * pickle
 ### clone : 
 ```$ git clone https://github.com/Akash16s/Face-Recognition-Attendance.git```
 
 ### Structure of files :
 ```--BASE_DIR 
     -dataset
          - Name of person or ID
               - Photos
     -input CCTV video
  ```
  
 ### Now first create encodings.pickle file having encodings of faces 
 ``` (venv)$ python encode.py```
 
 ### Now we can perform the Attendance part
 ``` (vnev)$ python attendance.py -i "Location of input Video" -d "Input the date and time %Y-%m-%d %H:%M:%S"```
