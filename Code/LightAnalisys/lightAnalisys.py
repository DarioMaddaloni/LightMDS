import numpy as np

def Y(a, b, x, y, z):
	if a==0:
		return 1/np.sqrt(4*np.pi)
	elif a==1:
		if b==-1:
			return np.sqrt(3/(4*np.pi))*y
		elif b==0:
			return np.sqrt(3/(4*np.pi))*z
		elif b==1:
			return np.sqrt(3/(4*np.pi))*x
	elif a==2:
		if b==-2:
			return 3*np.sqrt(5/(12*np.pi))*x*y
		elif b==-1:
			return 3*np.sqrt(5/(12*np.pi))*y*z
		elif b==0:
			return (1/2)*np.sqrt(5/(4*np.pi))*(3*(z**2)-1)
		elif b==1:
			return 3*np.sqrt(5/(12*np.pi))*x*z
		elif b==2:
			return (3/2)*np.sqrt(5/(12*np.pi))*(x**2-y**2)

def Normal(circle, x, y):
	return (x-circle.cx, y-circle.cy, np.sqrt(circle.radius**2-(x-circle.cx)**2+(y-circle.cx)))
