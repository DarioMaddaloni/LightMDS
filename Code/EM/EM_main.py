###   EXPECTATION MAXIMIZATION MAIN FILE

import sys
sys.path.insert(0, './../..')
#import os
#print(os.getcwd());	
from Code.import_file import *
from Code.circle_lib import *
from Code.show_lib import *
import expectation_lib as ex
import maximization_lib as ma


#Opening the image
imageName = "pallaPaint3soloCenterCenter"
image = cv2.imread("./../../Samples/EM/"+imageName+".png", 0)

currentRadius=70
currentCx=190
currentCy=186
currentCircle = circle(currentCx, currentCy, currentRadius)
threshold=1000

points = [] #qui ci sono tutti i punti della circonferenza stimata
allTheDk = []
allTheWk = []

#EXPECTATION
points = ex.updateOfThePointsOfBoundary(image, currentCircle, threshold)
for i in range(len(points)):
	allTheDk.append(ex.deltak(points[i][0], points[i][1], currentCircle))
currentEpsilon = ex.initializeEpsilon(image, currentCircle, threshold)
currentSigma = ex.initializeSigma(allTheDk)
for i in range(len(points)):
	allTheWk.append(ex.wk(allTheDk[i], currentSigma, currentEpsilon))

#MAXIMIZATION
M = ma.computeM(points)
W = ma.computeW(points, currentSigma, currentEpsilon, currentCircle)
v = ma.computeEigenvector(M, W)
currentCircle = ma.updateValues(v)


#for i in range(10): #qui ho messo una condizione a caso, ma sarà bene capire quando ferma expectation maximization
#	#Inizio la EXPECTATION
#	#Trovo i punti del bordo della circonferenza
#	points = ex.updateOfThePointsOfBoundary(image, currentCircle, threshold)
#	#Calcolo i deltak per ogni punto del bordo stimato
#	for i in range(len(points)):
#		allTheDk.append(ex.deltak(points[0], points[1], circleObj))
#	#Calcolo l'epsilon attraverso il numero di punti totali
#	currentEpsilon = ex.updateEpsilon(image, currentCircle, threshold)
#	#Calcolo la sigma come descritto nel paper
#	currentSigma = ex.updateSigma(allTheDk)
#	#Calcolo tutti i pesi di tutti i punti sul bordo stimato
#	for i in range(len(points)):
#		allTheWk.append(ex.wk(allTheDk[i], currentSigma, currentEpsilon))
#
#	#Inizio la MAXIMIZATION
#	#sigma = initializeSigma()
#	currentCircle.cx, currentCircle.cy, currentCircle.r = updateValues(v1,v2,v3,v4)

#Visualizing the image
show(ex.foundCircle(image, currentCircle, threshold))
show(image)

