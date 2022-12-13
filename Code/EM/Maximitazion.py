

def updateValues(v1, v2, v3, v4):
	return [updateCx(v1, v2), updateCy(v1, v3)], updateRadius(v1, v2, v3, v4)

def updateCx(v1, v2):
	return -v2/(2*v1)

def updateCy(v1, v3):
	return -v3/(2*v1);

def updateRadius(v1, v2, v3, v4):
	return np.sqrt((v2**2+v3**2)/(4*v1**2)) - v4/v1
