import numpy as np
from matplotlib import pyplot as plt

pixelLength = 1024
class point:
	def __init__(self, x:int, y:int):
		if (x<0) or (x>1024) or (y<0) or (y>1024):
			raise Exception("Point coordinates not in interval (0, {}}).".format(pixelLength))
		self.x = x
		self.y = y

	def isInImage(self):
		return not((self.x<0) or (self.x>1024) or (self.y<0) or (self.y>1024))
	def belongsToCircle(self, C):
		if not self.isInImage():
			raise Exception("Point coordinates not in interval (0, {}}).".format(pixelLength))
		return (self.x-C.center.x)**2 + (self.y-C.center.y)**2 < C.r**2

class circle:
	def __init__(self, cx: int, cy: int, r: int, sigma = 40, epsilon = 0.2): # constructor of the class
		self.center = point(cx,cy)
		self.r = r
		self.sigma = sigma
		self.epsilon = epsilon
		self.l00 = 0
		self.l1m1 = 0
		self.l10 = 0
		self.l11 = 0
		self.l2m2 = 0
		self.l2m1 = 0
		self.l20 = 0
		self.l21 = 0
		self.l22 = 0

	def __contains__(self, P:point):
		return (P.x-self.center.x)**2 + (P.y-self.center.y)**2 < self.r**2

	def normalAtPoint(self, P:point):# Returns the normal vector in the point P of the sphere
		if not P.belongsToCircle(self):
			raise Exception("The point does not belong to the circle.")
		else:
			n = np.zeros(3, dtype=float)
			n[0] = P.x-self.center.x
			n[1] =P.y-self.center.y
			n[2] = np.sqrt(self.r**2-(P.x-self.center.x)**2-(P.y-self.center.y)**2)
			return n / np.linalg.norm(n)

	def Y00(self, n):
		return 1/np.sqrt(4*np.pi)

	def Y1m1(self, n):
		return np.sqrt( 3/( 4*np.pi ) ) * n[1]

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
		for x in range(pixelLength):
			firstFound = False
			for y in range(pixelLength):
				P = point(x,y)
				if P in self:
					n = C.normalAtPoint(P)
					image[x,y] = self.l00*self.Y00(n)+		self.l1m1*(2*np.pi/3)*self.Y1m1(n)+self.l10*(2*np.pi/3)*self.Y10(n)+self.l11*(2*np.pi/3)*self.Y11(n)+		self.l2m2*(np.pi/4)*self.Y2m2(n)+self.l2m1*(np.pi/4)*self.Y2m1(n)+self.l20*(np.pi/4)*self.Y20(n)+self.l21*(np.pi/4)*self.Y21(n)+self.l22*(np.pi/4)*self.Y22(n)
					firstFound = True
				else: # Since spheres are convex figures, we can skip some iterations
					if firstFound:
						break
		return image

	def rendered(self):
		return self.renderedOnImage(np.zeros(shape = (pixelLength, pixelLength), dtype=float))

	def onImage(self, image, width = 2): # return a RGB image with the grayscale original image in background and the circle guess in red
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


if __name__ == '__main__':
	# qualche test
	C = circle(500,700,200)
	P = point(400, 720)
	n = C.normalAtPoint(P)
	print(C.Y00(n),	C.Y22(n),	C.Y1m1(n),	C.Y2m1(n),	C.Y2m2(n),	C.Y10(n),	C.Y11(n),	C.Y20(n),	C.Y21(n))

	C.l00=100
	C.l1m1 = 100
	C.l2m1 = 100
	image = C.rendered()
	image[:,1] = 0
	image[1,:] = 256
	print(image)
	plt.imshow(image,cmap='gray')
	plt.show()
