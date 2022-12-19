from Code.import_file import *
from Code.show_lib import *
from Code.circle_lib import *

def onclick(event):
        if event.xdata != None and event.ydata != None:
            global param
            param.append(event.xdata)
            param.append(event.ydata)

def guess(image):
    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(image)
    global param
    param = []
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Select the center of the image vith a first click and a point on the circunference with a second click.")
    plt.show()
    print(param)
    return circle(int(param[0]),int(param[1]),int(np.sqrt((param[0]-param[2])**2+(param[1]-param[3])**2)))


def norm(value, sigma):
	return scipy.stats.norm.pdf(value, loc=0, scale=sigma)

def deltak(xk, yk, circleObj):
	cx, cy, r = circleObj.cx, circleObj.cy, circleObj.r
	rk_quad = (xk-cx)**2 + (yk - cy)**2
	return np.abs( rk_quad - r**2 ), rk_quad

def wk(dk, sigma, p):
	value = norm(dk, sigma)
	epsilonBar = p
	epsilon = epsilonBar*np.sqrt(2*np.pi)*sigma
	return (value) / (value+epsilon)
#	value = norm(dk, sigma)
#	return (value) / (value+epsilon)
	
def computeP(image, circleObj):
	numberOfUniformPoints = counterOfTotalPoints(image) - expectedNumberOfCirclePoints(image, circleObj, circleObj.sigma)
	return numberOfUniformPoints / (image.shape[0]*image.shape[1])

def initializeSigma(allTheDk): #varianza di allTheDk
	mean = np.mean(allTheDk)
	sigma = 0
	return np.sqrt(np.mean([(deltak - mean)**2 for deltak in allTheDk]))

def updateSigma(allTheWk, allTheDk): #preso dal paper di Hany Farid
	allTheProduct=[];
	for i in range(len(allTheWk)):
		allTheProduct.append(allTheWk[i]*(allTheDk[i])**2)
	return np.sum(allTheProduct)/np.sum(allTheWk)
	
	
	

def counterOfTotalPoints(image): #da usare in initializeP ed updateP
	counter=0
	for i in image:
		for j in i:
			if j==255:
				counter=counter+1
	return counter

def totalPoints(image): #da usare in initializeP ed updateP
	points=[]
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			if image[xk][yk]==255:
				points.append([xk, yk])
	return points

def expectedNumberOfCirclePoints(image, circleObj, sigma):
	return np.sum([norm(deltak(i,j, circleObj), sigma) for i in range(image.shape[0]) for j in range(image.shape[1])])

def computeThreshold(percentageOfConfidenceInterval, sigma):
	return scipy.stats.norm.interval(percentageOfConfidenceInterval, loc=0, scale=sigma)[1]

def computePointsOfBoundary(image, circleObj, threshold): 
	points = []
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			if deltak(xk, yk, circleObj)<threshold and image[xk][yk]==255:	
				points.append([xk, yk]);
	return points

def counterOfCirclePoints(image, circleObj, threshold): #Sostituito da expectedNumberOfCirclePoints
	counter=0
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			if deltak(xk, yk, circleObj)<threshold and image[xk][yk]==255:	
				counter=counter+1
	return counter

def foundCircle(image, circleObj, threshold):
	fakeImage=image.copy()
	for xk in range(image.shape[0]):
		for yk in range(image.shape[1]):
			if deltak(xk, yk, circleObj)<threshold:
				fakeImage[xk][yk]=155
	return fakeImage


