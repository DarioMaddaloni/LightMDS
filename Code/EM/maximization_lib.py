from Code.import_file import *
from Code.circle_lib import *
import expectation_lib as ex

def computeM(pointsList):#takes as input the list of points [x1, y1], ... [xn, yn] fo the image that should be the edge fo the extimated circle
	n = len(pointsList) #the shape of our matrix M will be n x 4
	M = np.ones((n,4))# so that we do not need to place the ones in the last columns
	for i in range(n):
		M[i,0] = pointsList[i][0]**2 + pointsList[i][1]**2
		M[i,1] = pointsList[i][0]
		M[i,2] = pointsList[i][1]
	return M
	
def computeW(pointsList, sigma, epsilon, circleObj):
	return np.diag([ex.wk(   ex.deltak(point[0], point[1], circleObj) , sigma, epsilon) for point in pointsList])
	
	
	###N.B. durante la stesura di compute M e compute W mi sono reso conto che una possibile ottimizzazione del codice potrebbe essere utilizzare la matrice M stessa come spazio di storage per i valori delle coordinate contenuti in pointsList.. in pratica forse costruire M e W direttamente durate la fase di identificazione dei punti del bordo potrebbe essere la miglior strategia

def computeEigenvector(M, W):
	A = np.transpose(M) @ np.transpose(W) @ W @ M 
	eigValues, eigVectors = np.linalg.eig(A)
	return eigVectors[:, [np.argmin(eigValues)]]#np.argmin ci da la posizione ddell'autovalore minimo. Dovremmo forse considerare l'autovalore minore ma in modulo? In ogni caso l'uotput è un vettore n x 1'

def updateValues(v):#forse input sarà un vettore?
	return circle(updateCx(v[0], v[1]), updateCy(v[0], v[2]), updateRadius(v))#sostituire con oggetto della classe circle

def updateCx(v1, v2):
	return -v2/(2*v1)

def updateCy(v1, v3):
	return -v3/(2*v1);

def updateRadius(v):
	return np.sqrt((v[1]**2+v[2]**2)/(4*v[0]**2)) - v[3]/v[0]
	
	
