import cv2

class eraser:
	def __init__(self, screen, image, radius=20, eraser=False): #constructor of the class
		self.radius = radius
		self.eraser = eraser
		self.image = image
		self.screen = screen

	def draw_circle(self, x,y):
		# 'erase' circle
		cv2.circle(self.image, (x, y), self.radius, (0, 0, 0), -1)
		cv2.imshow(self.screen, self.image)

	def handleMouseEvent(self, event, x, y, flags, param):
		global eraser , radius
		if (event==cv2.EVENT_MOUSEMOVE):
			# update eraser position
			if self.eraser==True:
				self.draw_circle(x,y)
		elif event == cv2.EVENT_LBUTTONUP:
			# stop erasing
			self.eraser = False
		elif (event==cv2.EVENT_LBUTTONDOWN):
			# start erasing
			self.eraser=True
			self.draw_circle(x,y)
