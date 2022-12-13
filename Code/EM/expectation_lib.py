from Code.import_file import *
from Code.show_lib import *

def deltak(xk, yk, cx, cy, r):
	return np.abs( (xk-cx)**2 + (yk - cy)**2 - r**2 );

def wk(dk, sigma, epsilon):
	value = np.exp((-(dk**2))/(2*(sigma**2)))
	return (value) / (value+epsilon)
	
#def initializeEpsilon:

#def initializeSigma:
	
#def updateEpsilon:

def updateSigma(allTheWk, Allthedk):
	allTheProduct=[];
	for i in range(len(allTheWk)):
		allTheProduct[i]=allTheWk[i]*Allthedk[i]
	return np.sum(allTheProduct)/np.sum(allTheWk)

def counterOfTotalPoints(image):#da usare in initializeEpsilon ed updateEpsilon
	counter=0
	for i in image:
		for j in i:
			if j==255:
				counter=counter+1
	return counter

def counterOfCirclePoints(image, cx, cy, r, threshold):#da usare in initializeEpsilon ed updateEpsilon
	counter=0
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			delta=deltak(xk, yk, cx, cy, r)
			if delta<threshold and image[xk][yk]==255:	
				counter=counter+1
# Decomment if you want to see the representation of the circle
#			if delta<threshold:
#				image[xk][yk]=155
	return counter, image




