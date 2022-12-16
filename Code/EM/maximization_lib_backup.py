from Code.import_file import *
from Code.circle_lib import *
import expectation_lib as ex

def computeM(pointsList):#takes as input the list of points [x1, y1], ... [xn, yn] fo the image that should be the edge fo the extimated circle
	n = len(pointsList) #the shape of our matrix M will be n x 4
	#print(n)	
	M = np.ones((n,4))# so that we do not need to place the ones in the last columns
	for i in range(n):
		M[i,0] = pointsList[i][0]**2 + pointsList[i][1]**2
		M[i,1] = pointsList[i][0]
		M[i,2] = pointsList[i][1]
	return M
	
def computeW(allTheWk):
	return np.diag(allTheWk)
	
	
	###N.B. durante la stesura di compute M e compute W mi sono reso conto che una possibile ottimizzazione del codice potrebbe essere utilizzare la matrice M stessa come spazio di storage per i valori delle coordinate contenuti in pointsList.. in pratica forse costruire M e W direttamente durate la fase di identificazione dei punti del bordo potrebbe essere la miglior strategia

def computeEigenvector(M, W):
	A = np.transpose(M) @ np.transpose(W) @ W @ M 
	eigValues, eigVectors = np.linalg.eig(A)
#	print("eigValues = ", eigValues)
#	print("eigVectors = ", eigVectors)
	return eigVectors[:, [np.argmin(eigValues)]]#np.argmin ci da la posizione ddell'autovalore minimo. Dovremmo forse considerare l'autovalore minore ma in modulo? In ogni caso l'uotput è un vettore n x 1'

def updateValues(v):#forse input sarà un vettore?
	v = [v[i][0] for i in range(4)]
	return circle(updateCx(v[0], v[1]), updateCy(v[0], v[2]), updateRadius(v[0], v[1], v[2], v[3]))#sostituire con oggetto della classe circle

def updateCx(v1, v2):
	return -v2/(2*v1) #Ritorna una lista... Lunga uno con solo il valore..

def updateCy(v1, v3):
	return -v3/(2*v1);

def updateRadius(v1, v2, v3, v4):
	return np.sqrt((v2**2+v3**2)/(4*(v1**2)) - v4/v1)
	
	
