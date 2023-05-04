from Code.import_file import *
from Code.circle_lib import *


def computeEigenvector(M, W):
	A = np.transpose(M) @ np.transpose(W) @ W @ M 
	eigValues, eigVectors = np.linalg.eig(A)
#	print("eigValues = ", eigValues)
#	print("eigVectors = ", eigVectors)
	return eigVectors[:, [np.argmin(eigValues)]]#np.argmin ci da la posizione ddell'autovalore minimo. Dovremmo forse considerare l'autovalore minore ma in modulo? In ogni caso l'uotput è un vettore n x 1'


def updateCircle(C, v):#forse input sarà un vettore?
	v = [v[i][0] for i in range(4)]
	C.cx = updateCx(v[0], v[1])
	C.cy = updateCy(v[0], v[2])
	C.r = updateRadius(v[0], v[1], v[2], v[3])

def updateCx(v1, v2):
	return -v2/(2*v1) #Ritorna una lista... Lunga uno con solo il valore..

def updateCy(v1, v3):
	return -v3/(2*v1);

def updateRadius(v1, v2, v3, v4):
	return np.sqrt((v2**2+v3**2)/(4*(v1**2)) - v4/v1)
	
	
