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
currectCircle = circle(currentCx, currentCy, currentRadius)
threshold=1000
value = ex.counterOfCirclePoints(image, currectCircle, threshold)
epsilon = ex.initializeEpsilon(image, currectCircle, threshold)
#sigma = initializeSigma()

print(epsilon)
print(value)

#Visualizing the image
ex.foundCircle(image, currectCircle, threshold)

show(image);

