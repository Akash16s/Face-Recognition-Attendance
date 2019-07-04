import cv2
import math

class sceneChangeDetect:
	mean_previous = 0
	sum_frame = 0
	def detectChange(self,img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(5,5),0)
		edges = cv2.Canny(blur,100,150,apertureSize = 3)

		mean_now = edges.mean()

		self.sum_frame = (self.sum_frame+mean_now)/2 + 4

		sub = math.sqrt(abs(mean_now**4 - self.mean_previous**4) +100)
		#print("sub is "+str(sub))
		#print("sum is "+str(self.sum_frame))
		self.mean_previous=mean_now

		if(sub > self.sum_frame):
			return True
		else:
			return False