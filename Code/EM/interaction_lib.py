import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2
import math
from Code.circle_lib import circle




#image = cv2.imread("./../../Samples/DallE2/DallE2_{}.png".format(0), 0)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)



def tellme(s):
	plt.title(s, fontsize=16)
	plt.draw()


def guess(image):#guessing center and radius
		 
	plt.imshow(image)

	tellme('You will define a circle, click to begin')

	plt.waitforbuttonpress()

	while True:
		center = []
		while len(center) < 1:
			tellme('Select the center with mouse')
			center = np.asarray(plt.ginput(1, timeout=-1))
			if len(center) < 1:
				tellme('Too few points, starting over')
				time.sleep(1)  # Wait a second
				
		pts = []
		while len(pts) < 1:
			tellme('Select a point on the border')
			pts = np.asarray(plt.ginput(1, timeout=-1))
			print(pts)
			if len(pts) < 1:
				tellme('Too few points, starting over')
		
		r = math.dist(pts[0],center[0])
		
		C = matplotlib.patches.Circle((center[0][0],center[0][1]), radius=r, color='r',alpha = 0.4)
		
		
		fig = plt.gcf()
		ax = fig.gca()
		
		ax.add_patch(C)

		tellme('Happy? Key click for yes, mouse click for no')

		if plt.waitforbuttonpress():
			break
			
		C.remove()
	return circle(center[0][1],center[0][0],r)
	
	
   
   
   
   
#   guess3 is easier to use but we still do not handle exceptions (how to deal with non found intersections inside the image and vertical choosen points?) 
	
	
def guess3(image):#guessing 3 points in the borderof the circle 
		 
	plt.imshow(image)

	tellme('Click on the image to begin')

	plt.waitforbuttonpress()

	while True:

		#Select 3 point on the border and save their (x,y)-cohordinate in a list
		pts = []
		while len(pts) < 3:
			tellme('Select three points on the border')
			pts = np.asarray(plt.ginput(3, timeout=-1))
			print(pts)
			if len(pts) < 3:
				tellme('Too few points, starting over')

		
		intersection = []
		for x in range(image.shape[0]):
			#points in the first line
			y_1 = -(pts[1][0]-pts[0][0])/(pts[1][1]-pts[0][1])*(x-(pts[1][0]+pts[0][0])/2)+(pts[1][1]+pts[0][1])/2
			
			#points in the second line
			y_2 = -(pts[2][0]-pts[1][0])/(pts[2][1]-pts[1][1])*(x-(pts[2][0]+pts[1][0])/2)+(pts[2][1]+pts[1][1])/2
			
			
			if abs(y_1-y_2) < 5:
				intersection.append([x,y_1])
				intersection.append([x,y_2])

		center = np.mean(intersection, axis = 0)
		
		r = math.dist(pts[0],(center[0], center[1]))
		
		C = matplotlib.patches.Circle(center, radius=r, color='r',alpha = 0.4)
		
		
		fig = plt.gcf()
		ax = fig.gca()
		
		ax.add_patch(C)

		tellme('Happy? Key click for yes, mouse click for no')

		if plt.waitforbuttonpress():
			break
			
		C.remove()
	return circle(center[1],center[0],r)# inverted for no well understood reason, but properly working
	
