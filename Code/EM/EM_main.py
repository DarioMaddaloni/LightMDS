###   EXPECTATION MAXIMIZATION MAIN FILE

import sys
sys.path.insert(0, './../..')
#import os
#print(os.getcwd());	
from Code.import_file import *
from Code.circle_lib import *
from Code.show_lib import *
import expectation_lib as ex


#Opening the image
imageName = "pallaPaint3soloCenterCenter"
image = cv2.imread("./../../Samples/EM/"+imageName+".png", 0)

currentRadius=70
currentCx=190
currentCy=186
currentCircle = circle(currentCx, currentCy, currentRadius)
threshold=1000

points = ex.updateOfThePointsOfBoundary(image, currentCircle, threshold) #qui ci sono tutti i punti della circonferenza stimata

AlltheDk = []
AlltheWk = []


for i in range(10): #qui ho messo una condizione a caso, ma sarà bene capire quando ferma expectation maximization
	#Inizio la EXPECTATION
	#Calcolo i deltak per ogni punto del bordo stimato
	for i in range(len(points)):
		AlltheDk.append(ex.deltak(points[0], points[1], circleObj))
	#Calcolo l'epsilon attraverso il numero di punti totali
	currentEpsilon = ex.updateEpsilon(image, currentCircle, threshold)
	#Calcolo la sigma come descritto nel paper
	currentSigma = ex.updateSigma(AlltheDk)
	#Calcolo tutti i pesi di tutti i punti sul bordo stimato
	for i in range(len(points)):
		AlltheWk.append(ex.wk(AlltheDk[i], currentSigma, currentEpsilon))

	#Inizio la MAXIMIZATION
	#sigma = initializeSigma()
	currentCircle.cx, currentCircle.cy, currentCircle.r = updateValues(v1,v2,v3,v4)

#Visualizing the image
show(ex.foundCircle(image, currentCircle, threshold))
show(image)

