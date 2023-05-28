import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

pixelLength = 1024
class point:
	def __init__(self, x:int, y:int):
		if (x<0) or (x>1024) or (y<0) or (y>1024):
			raise Exception("Point coordinates not in interval (0, {}).".format(pixelLength))
		self.x = x
		self.y = y

	def isInImage(self):
		return not((self.x<0) or (self.x>1024) or (self.y<0) or (self.y>1024))

	def belongsToCircle(self, C):
		if not self.isInImage():
			raise Exception("Point coordinates not in interval (0, {}}).".format(pixelLength))
		return (self.x-C.center.x)**2 + (self.y-C.center.y)**2 < C.r**2

	def __repr__(self):
		return f"({self.x}, {self.y})"

	def onImage(self, image, width = 10): #return a RGB image with the grayscale original image in background and the circle guess in red
		if (len(image.shape) == 2):#grayscale image
			output = np.zeros((image.shape[0],image.shape[1],3), dtype = np.short)
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if image[i,j] == 255:#adding points of the image
						for k in range(3):
							output[i,j,k] = 255

			# Adding the point
			rad = int(np.ceil(width/2))
			for i in range(-rad, rad +1 ):
				for j in range(-rad, rad +1 ):
					output[self.x+i, self.y+j, 0] = 0
					output[self.x+i, self.y+j, 1] = 255
					output[self.x+i, self.y+j, 2] = 0
		else: #RGB image
			output = image

			# Adding the point
			rad = int(np.ceil(width / 2))
			for i in range(-rad, rad + 1):
				for j in range(-rad, rad + 1):
					output[self.x + i, self.y + j, 0] = 0
					output[self.x + i, self.y + j, 1] = 255
					output[self.x + i, self.y + j, 2] = 0

		return output

	@staticmethod
	def collectionOnImage(pointsList, image = np.zeros((pixelLength, pixelLength, 3))):
		"""Static method that prints many points on an image."""
		for i in range(len(pointsList)):
			image = pointsList[i].onImage(image)
		return image

class circle:
	def __init__(self, cx: int, cy: int, r: int, sigma = 40, epsilon = 0.2): # constructor of the class
		self.center = point(cx,cy)
		self.r = r
		self.sigma = sigma
		self.epsilon = epsilon

		# Coefficients, one coordinate for each layer, using only the 0-th coordinate for grayscale images
		self.l00 = np.zeros(shape = (3), dtype=float)
		self.l1m1 = np.zeros(shape = (3), dtype=float)
		self.l10 = np.zeros(shape = (3), dtype=float)
		self.l11 = np.zeros(shape = (3), dtype=float)
		self.l2m2 = np.zeros(shape = (3), dtype=float)
		self.l2m1 = np.zeros(shape = (3), dtype=float)
		self.l20 = np.zeros(shape = (3), dtype=float)
		self.l21 = np.zeros(shape = (3), dtype=float)
		self.l22 = np.zeros(shape = (3), dtype=float)

	def __repr__(self):
		return f"Circle with center ({self.center.x}, {self.center.y}) and radius {self.r}."

	def __contains__(self, P:point):
		return (P.x-self.center.x)**2 + (P.y-self.center.y)**2 < self.r**2

	def normalAtPoint(self, P:point):# Returns the normal vector in the point P of the sphere
		if not P.belongsToCircle(self):
			print("ciao");
			raise Exception("The point does not belong to the circle.")
		else:
			n = np.zeros(3, dtype=float)
			n[0] = P.x-self.center.x
			n[1] = P.y-self.center.y
			n[2] = np.sqrt(self.r**2-(n[0])**2-(n[1])**2)
			return n / np.linalg.norm(n)

	def Y00(self, n):
		return 1/np.sqrt( 4 * np.pi )

	def Y1m1(self, n):
		return np.sqrt(3 / ( 4 * np.pi ) ) * n[1]

	def Y10(self, n):
		return np.sqrt(3 / (4 * np.pi)) * n[2]

	def Y11(self, n):
		return np.sqrt(3 / (4 * np.pi)) * n[0]

	def Y2m2(self,n):
		return 3 *np.sqrt(5 / (12 * np.pi)) * n[0] * n[1]

	def Y2m1(self,n):
		return 3 *np.sqrt(5 / (12 * np.pi)) * n[1] * n[2]

	def Y20(self,n):
		return 0.5 * np.sqrt(5 / (4 * np.pi)) * (3*(n[2]**2)-1)

	def Y21(self,n):
		return 3 *np.sqrt(5 / (12 * np.pi)) * n[0] * n[2]

	def Y22(self,n):
		return 1.5 * np.sqrt(5 / (12 * np.pi)) * ((n[0]**2)-(n[1]**2))




	def renderedOnImage(self, image):
		""" Rendering the sphere on the image."""
		for x in range(pixelLength):
			firstFound = False # Flag useful to speed-up the algorithm
			for y in range(pixelLength):
				P = point(x,y)
				if P in self:
					firstFound = True
					n = self.normalAtPoint(P)

					match len(image.shape):
						case 3: # RGB image
							for i in range(3): # i cycling through the three RGB layers
								#n is the normal vector on the point P
								image[x, y, i] = self.l00[i] * np.pi * self.Y00(n) + \
												 self.l1m1[i] * (2 * np.pi / 3) * self.Y1m1(n) + \
												 self.l10[i] * (2 * np.pi / 3) * self.Y10(n) + \
												 self.l11[i] * ( 2 * np.pi / 3) * self.Y11(n) + \
												 self.l2m2[i] * (np.pi / 4) * self.Y2m2(n) + \
												 self.l2m1[i] * (np.pi / 4) * self.Y2m1(n) + \
												 self.l20[i] * (np.pi / 4) * self.Y20(n) + \
												 self.l21[i] * (np.pi / 4) * self.Y21(n) + \
												 self.l22[i] * (np.pi / 4) * self.Y22(n)
						case 2: # Grayscale image
							image[x, y] =	 self.l00[0] * np.pi * self.Y00(n) + \
											 self.l1m1[0] * (2 * np.pi / 3) * self.Y1m1(n) + \
											 self.l10[0] * (2 * np.pi / 3) * self.Y10(n) + \
											 self.l11[0] * (2 * np.pi / 3) * self.Y11(n) + \
											 self.l2m2[0] * (np.pi / 4) * self.Y2m2(n) + \
											 self.l2m1[0] * (np.pi / 4) * self.Y2m1(n) + \
											 self.l20[0] * (np.pi / 4) * self.Y20(n) + \
											 self.l21[0] * ( np.pi / 4) * self.Y21(n) + \
											 self.l22[0] * (np.pi / 4) * self.Y22(n)
						case other:
							raise Exception("The image where to render the sphere has not the correct format of an RGB or grayscale image.")
				else: # Since spheres are convex figures, we can skip some iterations
					if firstFound:
						break
		return image

	def fastRenderedOnImage(self, image):
		""" Rendering ball on image faster using precomputation and iterative procedure for finding points on image"""
		# Precomputing mu, i.e. the fixed multipliers involved in each pixel's value estimation
		mu = np.zeros((9), dtype=float)
		"""
		mu[0] = np.pi / np.sqrt(4 * np.pi)
		mu[1] = (2 * np.pi / 3) * np.sqrt(3 / (4 * np.pi))
		mu[2] = (2 * np.pi / 3) * np.sqrt(3 / (4 * np.pi))
		mu[3] = (2 * np.pi / 3) * np.sqrt(3 / (4 * np.pi))
		mu[4] = (np.pi / 4) * 3 * np.sqrt(5 / (12 * np.pi))
		mu[5] = (np.pi / 4) * 3 * np.sqrt(5 / (12 * np.pi))
		mu[6] = (np.pi / 4) * 0.5 * np.sqrt(5 / (4 * np.pi))
		mu[7] = (np.pi / 4) * 3 * np.sqrt(5 / (12 * np.pi))
		mu[8] = (np.pi / 4) * 1.5 * np.sqrt(5 / (12 * np.pi))
		"""

		mu[0] = np.pi / np.sqrt(4 * np.pi)
		mu[1] = (2 * np.pi / 3) * np.sqrt(3 / (4 * np.pi))
		mu[2] = mu[1]
		mu[3] = mu[1]
		val = (np.pi / 4) * np.sqrt(5 / (12 * np.pi))
		mu[4] = 3 * val
		mu[5] = mu[4]
		mu[6] = 0.5 * np.sqrt(3) * val
		mu[7] = mu[4]
		mu[8] = 1.5 * val
		start = time.time();

		# Iterate on all the points in the filling of the ball
		for xCoordinate in range(self.center.x-self.r + 1, self.center.x+1):
			d = int(np.floor(self.center.x - xCoordinate))
			c = int(np.floor(np.sqrt(self.r ** 2 - (d) ** 2)))
			addx = d
			#print(f"d = {d}")
			for yCoordinate in range(self.center.y-c , self.center.y+1):
				addy = int(np.floor(self.center.y - yCoordinate))

				try:
					n = self.normalAtPoint(point(xCoordinate, yCoordinate))
					n1 = np.array([n[0], -n[1], n[2]]) #xCoordinate1 = xCoordinate, yCoordinate1 = yCoordinate + 2*addy
					n2 = np.array([-n[0], n[1], n[2]]) #xCoordinate1 = xCoordinate + 2addx, yCoordinate = yCoordinate
					n3 = np.array([-n[0], -n[1], n[2]]) #xCoordinate1 = xCoordinate + 2addx, yCoordinate = yCoordinate + 2*addy
				except:
					continue # Catching exception in the case the point is not on the ball

				# Computing normal-dependent parameters
				Y = np.zeros((9), dtype=float)
				Y[0] = mu[0]
				Y[2] = mu[2] * n[2]
				Y[6] = mu[6] * (3 * (n[2] ** 2) - 1)
				Y[8] = mu[8] * ((n[0] ** 2) - (n[1] ** 2))

				Y[1] = mu[1] * n[1]
				Y[3] = mu[3] * n[0]
				Y[4] = mu[4] * n[0] * n[1]
				Y[5] = mu[5] * n[1] * n[2]
				Y[7] = mu[7] * n[0] * n[2]
				
				match len(image.shape):
					case 3:  # RGB image
						for i in range(3):  # i cycling through the three RGB layers
							val0123 =	 self.l00[i] * Y[0] + \
										 self.l10[i] * Y[2] + \
										 self.l20[i] * Y[6] + \
										 self.l22[i] * Y[8]
							val01 =	 self.l11[i] * Y[3] + \
									 self.l21[i] * Y[7]
							val23 = -val01
							val03 =	 self.l2m2[i] * Y[4]
							val12 = -val03
							val02 =	 self.l1m1[i] * Y[1] + \
									 self.l2m1[i] * Y[5]
							val13 = -val02
							
							image[xCoordinate, yCoordinate, i] = val0123 + val01 + val03 + val02
							image[xCoordinate, yCoordinate+2*addy, i] = val0123 + val01 + val12 + val13
							image[xCoordinate+2*addx, yCoordinate, i] = val0123 + val23 + val12 + val02
							image[xCoordinate+2*addx, yCoordinate+2*addy, i] = val0123 + val23 + val03 + val13

					case 2:  # Grayscale image
							val0123 =	 self.l00[i] * Y[0] + \
										 self.l10[i] * Y[2] + \
										 self.l20[i] * Y[6] + \
										 self.l22[i] * Y[8]
							val01 =	 self.l11[i] * Y[3] + \
									 self.l21[i] * Y[7]
							val23 = -val01
							val03 =	 self.l2m2[i] * Y[4]
							val12 = -val03
							val02 =	 self.l1m1[i] * Y[1] + \
									 self.l2m1[i] * Y[5]
							val13 = -val02
							
							image[xCoordinate, yCoordinate] = val0123 + val01 + val03 + val02
							image[xCoordinate, yCoordinate+2*addy] = val0123 + val01 + val12 + val13
							image[xCoordinate+2*addx, yCoordinate] = val0123 + val23 + val12 + val02
							image[xCoordinate+2*addx, yCoordinate+2*addy] = val0123 + val23 + val03 + val13
					case other:
						raise Exception(
							"The image where to render the sphere has not the correct format of an RGB or grayscale image.")
		end = time.time();
		print(f"time for long cycle {end - start} s")
		return image

	def rendered(self):
		return self.fastRenderedOnImage(np.zeros(shape = (pixelLength, pixelLength, 3), dtype=np.uint8))

	def grayscaleRendered(self):
		return self.fastRenderedOnImage(np.zeros(shape = (pixelLength, pixelLength), dtype=np.uint8))

	def onImage(self, image, width = 2): #return a RGB image with the grayscale original image in background and the circle guess in red
		if (len(image.shape) == 2):#grayscale image
			output = np.zeros((image.shape[0],image.shape[1],3), dtype = np.short)
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if image[i,j] == 255:#adding points of the image
						for k in range(3):
							output[i,j,k] = 255
					if (abs(np.sqrt((i-self.center.x)**2+(j-self.center.y)**2)-self.r) < width):#adding points of the circle
						output[i,j,0] = 0
						output[i,j,1] = 255
						output[i,j,2] = 0
		else: #RGB image
			output = image
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if (abs(np.sqrt((i-self.center.x)**2+(j-self.center.y)**2)-self.r) < width):#adding points of the circle
						output[i,j,0] = 0
						output[i,j,1] = 255
						output[i,j,2] = 0
		return output

	def randomPoint(self, N):
		"Extracts a list of N points at random in the filling of the ball."

		# Number of possible points
		maxNumber = int(np.floor(2*self.r + 4 * np.sum([np.sqrt(self.r**2 - i**2) for i in range(1, self.r)])))

		# Points list
		pointsList = []

		for _ in range(N):
			flag = True
			while flag: # Some points may be outside the image borders
				# Select at random the "index" of the point
				index = np.random.randint(0, maxNumber)

				# Compute the bijection and find the point
				s = 0 # Actual sum
				for xIndex in range(-self.r + 1, self.r):
					l = 2 * int(np.floor(np.sqrt(self.r**2 - xIndex**2)))
					s += l
					if s > index:
						s -= l
						yIndex = index-s - int(np.floor(l/2))
						if (self.center.x+xIndex in range(0, pixelLength)) and (self.center.y+yIndex in range(0,pixelLength)):
							pointsList.append(point(x= self.center.x+xIndex, y= self.center.y+yIndex))
							flag = False
						else:
							pass # In that case we must repeat another while iteration because the point is not inside the image borders
						break

		return pointsList

	def extimateCoefficients(self, image, N = 9):
		"""Extimate the rendering coefficients of the sphere using N points."""

		# Constructiong the list of the points
		pointsList = self.randomPoint(N)

		# Constructing the matrix A
		A = []
		for p in pointsList:
			n = self.normalAtPoint(p)
			A.append([ np.pi * self.Y00(n), 			(2 * np.pi / 3) * self.Y1m1(n), 	(2 * np.pi / 3) * self.Y10(n),
					   (2 * np.pi / 3) * self.Y11(n), 	(np.pi / 4) * self.Y2m2(n), 		(np.pi / 4) * self.Y2m1(n),
					   (np.pi / 4) * self.Y20(n), 		(np.pi / 4) * self.Y21(n), 			(np.pi / 4) * self.Y22(n)]
					)
		A = np.array(A)


		match len(image.shape):
			case 3: # RGB image
				for i in range(3):  # i cycling through the three RGB layers

					# Constructing the vector b for each layer
					b = np.array([image[p.x, p.y, i] for p in pointsList])

					# Solving the system
					l = np.linalg.lstsq(A, b, rcond=None)[0]

					# Storing the results
					self.l00[i] = l[0]
					self.l1m1[i] = l[1]
					self.l10[i] = l[2]
					self.l11[i] = l[3]
					self.l2m2[i] = l[4]
					self.l2m1[i] = l[5]
					self.l20[i] = l[6]
					self.l21[i] = l[7]
					self.l22[i] = l[8]

			case 2:  # Grayscale image

				# Constructing the vector b
				b = np.array([image[p.x, p.y] for p in pointsList])

				# Solving the system
				l = np.linalg.lstsq(A, b, rcond= None)[0]

				# Storing the results
				self.l00[0] = l[0]
				self.l1m1[0] = l[1]
				self.l10[0] = l[2]
				self.l11[0] = l[3]
				self.l2m2[0] = l[4]
				self.l2m1[0] = l[5]
				self.l20[0] = l[6]
				self.l21[0] = l[7]
				self.l22[0] = l[8]

			case other:
				raise Exception(
					"The image where to render the sphere has not the correct format of an RGB or grayscale image.")


if __name__ == '__main__':
	# Tests

	"""Testing random point extraction on the filling"""
	"""C = circle(500,700,200)
	pointsList = C.randomPoint(15)
	print(pointsList)
	IM = point.collectionOnImage(pointsList)
	IM = C.onImage(IM)
	plt.imshow(IM, vmin= 0, vmax=255)
	plt.show()

	# Testing rendering
	P = point(400, 720)
	n = C.normalAtPoint(P)
	print(C.Y00(n),	C.Y22(n),	C.Y1m1(n),	C.Y2m1(n),	C.Y2m2(n),	C.Y10(n),	C.Y11(n),	C.Y20(n),	C.Y21(n))

	# RGB test
	C.l00=np.array([10,0,10])
	C.l1m1 = np.array([6,0,6])
	C.l2m1 = np.array([4,0,4])
	image = C.rendered()
	plt.imshow(image, vmin=0, vmax=255)
	plt.show()

	# Grayscale test
	C.l00 = np.array([10, 0, 0])
	C.l1m1 = np.array([20, 0, 0])
	C.l2m1 = np.array([50, 0, 0])
	image = C.grayscaleRendered()
	plt.imshow(image ,cmap='gray', vmin=0, vmax=255)
	plt.show()"""

	"""Testing estimation of coefficients"""
	import time

	# Opening the image and defining a circle (values for center and radius chosen carefully)
	originalImage = np.asarray(Image.open("./../Samples/DallE2/DallE2_1.png"), dtype=np.uint8)
	C = circle(587, 432, 301)

	# Estimating the coefficients
	start = time.time()
	print("Estimating coefficients...")
	C.extimateCoefficients(originalImage, N=150)
	end = time.time()
	print(f"Extimated coefficients in {end - start} s")




	# Plotting original and rendered images comparing fast and previous algorithm for rendering

	matplotlib.rcParams['figure.figsize'] = [25, 25]

	plt.subplot(121)
	plt.title('Original image')
	plt.imshow(originalImage)
	"""
	plt.subplot(132)
	plt.title('Rendered ball on image')
	print("Rendering ball on image...")
	start = time.time()
	rendered = C.renderedOnImage(originalImage)
	end = time.time()
	print(f"Image rendered in {end - start} s")
	plt.imshow(rendered)
	"""
	plt.subplot(122)
	print("Fast rendering image...")
	plt.title('Rendered image in black background')
	start = time.time()
	fastRendered = C.fastRenderedOnImage(originalImage)
	end = time.time()
	print(f"Image fast rendered in {end - start} s")
	plt.imshow(fastRendered)
	plt.show()
