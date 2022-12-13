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

currentRadius=70
currentCx=190
currentCy=186
xk=120
yk=180
value, image=ex.counterOfCirclePoints(image, currentCx, currentCy, currentRadius, 1000)

print(value);

#Visualizing the image
show(image)
