import sys

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, './Code/EM')
from eraser_lib import eraser
import expectation_lib as ex
import maximization_lib as ma
from EM_main import EM
from interaction_lib import interactiveGuess
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image
sys.path.insert(0, './')
from Code.import_file import *
from Code.LightAnalisys.geometry_lib import circle
from Code.show_lib import *

if __name__ == "__main__":  # Execute a test in the case the algorithm is executed as a script
	import time
	for i in range(1, 10):  # loop in the DallE2-generated database
	
		print("\n#####	    Image {}".format(i) + ".\n      #####")
		image = np.asarray(Image.open("./Samples/DallE2/DallE2_{}.png".format(i)), dtype=np.uint8)

		C = interactiveGuess(image)
		print(C)
		rounds = 10

		C = EM(image, C, rounds = 10, visual=0, erase=1)
		print("cx = " + str(C.center.x))
		print("cy = " + str(C.center.y))
		print("r = " + str(C.r))
		print("sigma = " + str(C.sigma))
		print("epsilon = " + str(C.epsilon))
		C = circle(int(np.floor(C.center.x)), int(np.floor(C.center.y)), int(np.floor(C.r)), C.sigma, C.epsilon)

		# Plotting original and rendered images comparing fast and previous algorithm for rendering

		matplotlib.rcParams['figure.figsize'] = [25, 25]

		plt.subplot(131)
		plt.title('Original image')
		plt.imshow(image)

		plt.subplot(132)
		result = ndimage.median_filter(np.asarray(Image.open("./Samples/DallE2/DallE2_{}.png".format(i)), dtype=np.uint8), size=9)
		plt.imshow(result)
		
		# Estimating the coefficients with median_filter
		start = time.time()
		print("Estimating coefficients...")
		C.extimateCoefficients(np.asarray(Image.open("./Samples/DallE2/DallE2_{}.png".format(i)), dtype=np.uint8), N=200)
		end = time.time()
		print(f"Extimated coefficients in {end - start} s")

		# Plotting original and rendered images comparing fast and previous algorithm for rendering
		plt.subplot(133)
		print("Fast rendering image...")
		plt.title('Rendered image in black background')
		start = time.time()
		fastRendered = C.fastRenderedOnImage(image)
		end = time.time()
		print(f"Image fast rendered in {end - start} s")
		plt.imshow(fastRendered)
		plt.show()
