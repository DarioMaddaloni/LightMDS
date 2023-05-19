###	 EXPECTATION MAXIMIZATION MAIN FILE

import sys

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, './../..')
from Code.import_file import *
from Code.geometry_lib import circle
from Code.eraser_lib import eraser
from Code.show_lib import *
import expectation_lib as ex
import maximization_lib as ma
from interaction_lib import interactiveGuess
from scipy.ndimage import gaussian_filter
from PIL import Image


def preprocessing(originalImage, erase):
	print("\n --- START OF PREPROCESSING --- ")

	# Converting the image from RGB to grayscale
	print("RGB to grayscale...")
	if len(originalImage.shape) == 3:  # In that case we are analyzing an RGB images
		image = cv2.cvtColor(originalImage, cv2.COLOR_RGB2GRAY)  # Converting the image to grayscale
	else:
		assert (len(originalImage.shape) == 2)  # In that case we expect the image to be already in grayscale format
		image = originalImage

	# Sharpening
	print("Sharpening...")
	sigma = 0.12
	alpha = 3
	filter_blurred_f = gaussian_filter(image, sigma)
	image = image + alpha * (image - filter_blurred_f)

	# Edge detection and threshold
	print("Blurring...")
	image = cv2.blur(image, (11, 11))
	print("Cannying...")  # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
	image = cv2.Canny(image, threshold1=30, threshold2=70)

	print("Grayscale to black or white...")
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i, j] > 125:
				image[i, j] = 255
			else:
				image[i, j] = 0

	if erase:  # Erase noise by hand
		print("Eraser...")

		# Call eraser methods
		e = eraser( image, alwaysRefresh=0)
		e.connect()
		e.show()

		# Obtain the resulting image
		image = e.close()

	print(" --- END OF PREPROCESSING --- \n")

	return image


def EM(originalImage, C=0, rounds=0, visual=0, finalVisual=1, erase=1):
	"""
		originalImage := the image where to find the circle
		C := the  guess of the circle object
		rounds := the desired number of rounds
		visual := run the function showing (1) or not showing the prints in each step of the algorithm
		erase := the optional usage of the eraser to delete by hand points related to outliers
	"""

	# Preprocessing
	image = preprocessing(originalImage, erase)

	print("\n --- START OF E.M. ALGORITHM --- ")

	# Setting the circle guess in case it is not defined
	if C == 0:  # if we have no initial guess we start from the circle centered in the center of the image and with radious 1/3 of the smallest edge of the image
		print("Setting auto guess...")
		C = circle(image.shape[0] / 2, image.shape[1] / 2, min(image.shape) / 3, sigma=80, epsilon=0.3)

	if visual:
		# Plotting the processed image also displaying the initial guess
		print("Showing processed image with initial circle guess...")
		matplotlib.rcParams['figure.figsize'] = [15, 15]
		plt.title('Processed image with initial circle guess.')
		plt.imshow(C.onImage(image))
		plt.show()

	# Speedup condition parameters
	max_range = 100
	sigma_range = 25

	for round in range(1, rounds + 1):
		print("\nExecute round {}/{} with parameters:\nEpsilon = {}       Sigma = {}      Max Range = {}".format(round,
																											   rounds,
																											   C.epsilon,
																											   C.sigma,
																											   max_range))

		if visual:  # Plotting the actual probability curve depending on the distance from the circle edge
			ex.plot_prob_curve_evo(C.sigma, C.epsilon)

		M = []
		sigma_num = 0
		sigma_den = 0
		w = []

		# cycling in the image pixels in order to compute:
		#	 the values delta_k for each pixel of the image representing one (or 255) (stored in dk_1)
		#	 the matrix M
		#	 the list of the probabilities w_k
		# 	 the new valiue for the patrameter sigma
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				if image[i, j] == 255 and (
						dk := ex.deltak_evo(i, j,
											C)) < max_range ** 2:  # The second condition provides speedup
					#
					M.append([i ** 2 + j ** 2, i, j, 1])
					w.append(ex.wk(dk, C.sigma, C.epsilon))
					if dk < sigma_range ** 2:  # Second speedup condition
						# Calculations regarding the update of the parameter sigma
						wk_s = ex.wk_sigma(dk, C.sigma, C.epsilon)
						sigma_num += wk_s * dk
						sigma_den += wk_s

		# Updating the actual guess
		W = np.diag(np.array(w))
		v = ma.computeEigenvector(M, W)
		ma.updateCircle(C, v)

		# Updating the parameter sigma
		oldSigma = C.sigma
		C.sigma = sigma_num / sigma_den

		if abs(
				oldSigma - C.sigma) < 0.5:
			print("\nA small sigma update has happened. Iteration stopped." )
			break  # Breaking the for loop if there is no relevant change for the parameter sigma

		if visual:
			# Visualize the actual guess
			print("Swowing the actual guess...")
			matplotlib.rcParams['figure.figsize'] = [7, 7]
			plt.title('Circle estimation after {} step.'.format(round))
			plt.imshow(C.onImage(image))
			plt.show()

	if finalVisual:
		# Visualize the final guess
		print("Swowing the final guess...")
		matplotlib.rcParams['figure.figsize'] = [7, 7]
		plt.title('Final estimation.')
		plt.imshow(C.onImage(originalImage))
		plt.show()

	print("Final guess returned.")
	print("\n --- END OF E.M. ALGORITHM --- ")
	return C


if __name__ == "__main__":  # Execute a test in the case the algorithm is executed as a script
	for i in range(1, 10):  # loop in the DallE2-generated database
		print("\n#####	    Image {}".format(i) + ".\n      #####")
		image = np.asarray(Image.open("./../../Samples/DallE2/DallE2_{}.png".format(i)), dtype=np.uint8)

		C = interactiveGuess(image)
		print(C)
		rounds = 10

		EM(image, C, rounds = 10, visual=1, erase=1)


