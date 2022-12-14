from Code.import_file import *
from Code.show_lib import *
from Code.circle_lib import *

def deltak(xk, yk, circleObj):
	cx, cy, r = circleObj.cx, circleObj.cy, circleObj.r
	return np.abs( (xk-cx)**2 + (yk - cy)**2 - r**2 )

def wk(dk, sigma, epsilon):
	value = np.exp((-(dk**2))/(2*(sigma**2)))
	return (value) / (value+epsilon)
	
def initializeEpsilon(image, circleObj, threshold):
	return 1 - (counterOfCirclePoints(image, circleObj, threshold))/(counterOfTotalPoints(image))

def initializeSigma(allTheDk):
	mean=np.sum(allTheDk)/len(allTheDk)
	total=0
	for i in range(len(allTheDk)):
		total=total+(allTheDk[i]*mean)**2
	return np.sqrt(total)/2

def updateEpsilon(image, circleObj, threshold):
	return initializeEpsilon(image, circleObj, threshold)

def updateSigma(allTheWk, allTheDk):
	allTheProduct=[];
	for i in range(len(allTheWk)):
		allTheProduct[i]=allTheWk[i]*allTheDk[i]
	return np.sum(allTheProduct)/np.sum(allTheWk)

def counterOfTotalPoints(image):#da usare in initializeEpsilon ed updateEpsilon
	counter=0
	for i in image:
		for j in i:
			if j==255:
				counter=counter+1
	return counter

def counterOfCirclePoints(image, circleObj, threshold):#da usare in initializeEpsilon ed updateEpsilon
	counter=0
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			delta=deltak(xk, yk, circleObj)
			if delta<threshold and image[xk][yk]==255:	
				counter=counter+1
	return counter

def updateOfThePointsOfBoundary(image, circleObj, threshold):
	points = []
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			delta=deltak(xk, yk, circleObj)
			if delta<threshold and image[xk][yk]==255:	
				points.append([xk, yk]);
	return points

def foundCircle(image, circleObj, threshold):
	fakeImage=image.copy()
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			delta=deltak(xk, yk, circleObj)
			if delta<threshold:
				fakeImage[xk][yk]=155
	return fakeImage


