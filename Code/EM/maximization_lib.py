def computeM

def computeW:

def computeEigenvector:

def updateValues(v1, v2, v3, v4):#forse input sarà un vettore?
	return [updateCx(v1, v2), updateCy(v1, v3)], updateRadius(v1, v2, v3, v4)#sostituire con oggetto della classe circle

def updateCx(v1, v2):
	return -v2/(2*v1)

def updateCy(v1, v3):
	return -v3/(2*v1);

def updateRadius(v1, v2, v3, v4):
	return np.sqrt((v2**2+v3**2)/(4*v1**2)) - v4/v1
