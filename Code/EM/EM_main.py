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
images = ["linearNoise.png" , "pallaPaint2ExternalNoise.png", "pallaPaint3LinearNoise.png" ,  "pallaPaint3soloTopLeft.png" , "pallaPaint4soloCenterRight.png", "pallaPaint1soloDownLeft.png"  ,  "pallaPaint3ExternalNoise.png"  ,"pallaPaint3soloCenterCenter.png" , "pallaPaint3soloTopRight.png" , "randomNoise.png", "pallaPaint1soloTopCenter.png" , "pallaPaint3InternalNoise.png" , "pallaPaint3soloDownRight.png" ,"pallaPaint4soloCenterCenter.png"]

for imageName in images:
	image = cv2.imread("./../../Samples/EM/"+imageName, 0)

	currentCx=200 #180
	currentCy=200 #200
	currentRadius=90 #90
	currentCircle = circle(currentCx, currentCy, currentRadius)
	threshold=100000
	show(ex.foundCircle(image, currentCircle, 200))

	points = [] #qui ci sono tutti i punti della circonferenza stimata
	allTheDk = []
	allTheWk = []

	#EXPECTATION
	points = ex.updateOfThePointsOfBoundary(image, currentCircle, threshold)

	for i in range(len(points)):
		allTheDk.append(ex.deltak(points[i][0], points[i][1], currentCircle))

	currentEpsilon = ex.initializeEpsilon(image, currentCircle, threshold)

	currentSigma = 3000 #ex.initializeSigma(allTheDk)
	#print("sigma = ", currentSigma)

	for i in range(len(points)):
		allTheWk.append(ex.wk(allTheDk[i], currentSigma, currentEpsilon))

	#MAXIMIZATION
	#print("\nnumeratore di wk = ", np.exp((-(allTheDk[1]**2))/(2*(currentSigma**2))))
	#print("\nepsilon = ",currentEpsilon)
	M = ma.computeM(points)
	#print("\nM:\n", M)
	W = ma.computeW(allTheWk)
	#print("\nall the wk: \n", allTheWk)
	v = ma.computeEigenvector(M, W)
	#show(ex.foundCircle(image, currentCircle, threshold))
	currentCircle = ma.updateValues(v)

	print("\nnuovo cerchio di parametri:\n", currentCircle.cx)
	print(currentCircle.cy)
	print(currentCircle.r)
	#show(ex.foundCircle(image, currentCircle, threshold))

	#Visualizing the image
#	show(ex.foundCircle(image, currentCircle, 200))
	#show(image)

	for u in range(5):
		#EXPECTATION
		points = ex.updateOfThePointsOfBoundary(image, currentCircle, threshold)
		
		allTheDk = []
		for i in range(len(points)):
			allTheDk.append(ex.deltak(points[i][0], points[i][1], currentCircle))

		currentEpsilon = ex.initializeEpsilon(image, currentCircle, threshold)

		currentSigma = 3000 #ex.initializeSigma(allTheDk)
		#print("sigma = ", currentSigma)
		
		allTheWk = []
		for i in range(len(points)):
			allTheWk.append(ex.wk(allTheDk[i], currentSigma, currentEpsilon))

		#MAXIMIZATION
		#print("\nnumeratore di wk = ", np.exp((-(allTheDk[1]**2))/(2*(currentSigma**2))))
		#print("\nepsilon = ",currentEpsilon)
		M = ma.computeM(points)
		print("M shape = ", M.shape)
		#print("\nM:\n", M)
		W = ma.computeW(allTheWk)
		print("W shape = ", W.shape)
		#print("\nall the wk: \n", allTheWk)
		v = ma.computeEigenvector(M, W)
		#show(ex.foundCircle(image, currentCircle, threshold))
		currentCircle = ma.updateValues(v)

		print("\nnuovo cerchio di parametri:\n", currentCircle.cx)
		print(currentCircle.cy)
		print(currentCircle.r)

		#Visualizing the image
	#	show(ex.foundCircle(image, currentCircle, 200))
		#show(image)

	show(ex.foundCircle(image, currentCircle, 200))
