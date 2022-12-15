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

listOfImages = []
#Opening the image
images = ["pallaPaint4soloCenterRight.png"]#, "pallaPaint2ExternalNoise.png", "pallaPaint3LinearNoise.png" ,  "pallaPaint3soloTopLeft.png" , "pallaPaint4soloCenterRight.png", "pallaPaint1soloDownLeft.png",  "pallaPaint3ExternalNoise.png"  ,"pallaPaint3soloCenterCenter.png" , "pallaPaint3soloTopRight.png" , "randomNoise.png", "pallaPaint1soloTopCenter.png" , "pallaPaint3InternalNoise.png" , "pallaPaint3soloDownRight.png" ,"pallaPaint4soloCenterCenter.png", "linearNoise.png"]

for imageName in images:
	print(imageName)
	image = cv2.imread("./../../Samples/EM/"+imageName, 0)

	currentCx=200 #180
	currentCy=200 #200
	currentRadius=90 #90
	currentCircle = circle(currentCx, currentCy, currentRadius)
	show(ex.foundCircle(image, currentCircle, 200))

	points = [] #qui ci sono tutti i punti della circonferenza stimata
	allTheDk = []
	allTheWk = []

	#EXPECTATION
	points = ex.totalPoints(image)

	for i in range(len(points)):
		allTheDk.append(ex.deltak(points[i][0], points[i][1], currentCircle))

	currentSigma = ex.initializeSigma(allTheDk) #3000
	#print("sigma = ", currentSigma)
	currentP = ex.computeP(image, currentCircle, currentSigma)
	threshold=ex.computeThreshold(0.8, currentSigma)
	for i in range(len(points)):
		allTheWk.append(ex.wk(allTheDk[i], currentSigma, currentP))

	#MAXIMIZATION
	#print("\nnumeratore di wk = ", np.exp((-(allTheDk[1]**2))/(2*(currentSigma**2))))
	#print("\nepsilon = ",currentP)
	M = ma.computeM(points)
	#print("\nM:\n", M)
	W = ma.computeW(allTheWk)
	#print("\nall the wk: \n", allTheWk)
	v = ma.computeEigenvector(M, W)
	#show(ex.foundCircle(image, currentCircle, threshold))
	currentCircle = ma.updateValues(v)
	
	#print("\nSigma\n:", currentSigma)
	#print("\nnuovo cerchio di parametri:\n", currentCircle.cx)
	#print(currentCircle.cy)
	#print(currentCircle.r)
	#show(ex.foundCircle(image, currentCircle, 200))

	#Visualizing the image
	#show(ex.foundCircle(image, currentCircle, 200))
	#show(image)
	
	for u in range(3):
		print(u)
		points = [] #qui ci sono tutti i punti della circonferenza stimata
		

		#EXPECTATION
		points = ex.computePointsOfBoundary(image, currentCircle, threshold)

		currentSigma = ex.initializeSigma(allTheDk) #3000
		allTheDk = []
		allTheWk = []
		for i in range(len(points)):
			allTheDk.append(ex.deltak(points[i][0], points[i][1], currentCircle))
		#print("sigma = ", currentSigma)
		currentP = ex.computeP(image, currentCircle, currentSigma)
		threshold=ex.computeThreshold(0.8, currentSigma)
		for i in range(len(points)):
			allTheWk.append(ex.wk(allTheDk[i], currentSigma, currentP))

		#MAXIMIZATION
		#print("\nnumeratore di wk = ", np.exp((-(allTheDk[1]**2))/(2*(currentSigma**2))))
		#print("\nepsilon = ",currentP)
		M = ma.computeM(points)
		#print("\nM:\n", M)
		W = ma.computeW(allTheWk)
		#print("\nall the wk: \n", allTheWk)
		v = ma.computeEigenvector(M, W)
		#show(ex.foundCircle(image, currentCircle, threshold))
		currentCircle = ma.updateValues(v)
		
		#print("\nSigma\n:", currentSigma)
		#print("\nnuovo cerchio di parametri:\n", currentCircle.cx)
		#print(currentCircle.cy)
		#print(currentCircle.r)
		#show(ex.foundCircle(image, currentCircle, 200))


		#Visualizing the image
		#show(ex.foundCircle(image, currentCircle, 200))
		#show(image)
		
	listOfImages.append(ex.foundCircle(image, currentCircle, 200))
#
#ua sto rappresentando tutte le immagini con la nostra stima in un solo plot
xFigure = np.floor(np.sqrt(len(images)))
yFigure = np.ceil(np.sqrt(len(images)))
plt.figure(figsize=(15, 6))
for i in range(len(listOfImages)):
	plt.subplot(xFigure, yFigure, i+1)
	plt.title("Image: " + images[i])
	plt.imshow(listOfImages[i],cmap='gray')
plt.show()
