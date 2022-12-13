###   EXPECTATION MAXIMIZATION MAIN FILE

import sys
sys.path.insert(0, './../..')
#import os
#print(os.getcwd());	
from Code.import_file import *
from Code.show_lib import *
import Expectation as ex


#Opening the image
imageName = "pallaPaint3soloCenterCenter"
image = cv2.imread("./../../Samples/EM/"+imageName+".png", 0)

currentRadius=60
currentCx=200
currentCy=180
xk=120
yk=180
print(ex.deltak(xk, yk, currentCx, currentCy, currentRadius))
print(ex.counterOfTotalPoints(image))


#Visualizing the image
show(image)
