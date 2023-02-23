import numpy as np


class circle:
	def __init__(self, cx, cy, r, sigma = 3000): # constructor of the class
		self.cx = cx
		self.cy = cy
		self.r = r
		self.sigma = sigma

	def onImage(self, image, width = 2): # return a RGB image with the grayscale original image in background and the circle guess in red
		if (len(image.shape) == 2):#grayscale image
			output = np.zeros((image.shape[0],image.shape[1],3), dtype = np.short)
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if image[i,j] == 255:#adding points of the image
						for k in range(3):
							output[i,j,k] = 255
					if (abs(np.sqrt((i-self.cx)**2+(j-self.cy)**2)-self.r) < width):#adding points of the circle
						output[i,j,0] = 255
						output[i,j,1] = 0
						output[i,j,2] = 0
		else:#RGB image
			output = image
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if (abs(np.sqrt((i-self.cx)**2+(j-self.cy)**2)-self.r) < width):#adding points of the circle
						output[i,j,0] = 255
						output[i,j,1] = 0
						output[i,j,2] = 0
		return output 
