###   EXPECTATION MAXIMIZATION MAIN FILE

import sys
sys.path.insert(0, './../..')
#import os
#print(os.getcwd());	
from Code.import_file import *
from Code.circle_lib import circle
from Code.show_lib import *
import expectation_lib as ex
import maximization_lib as ma

listOfImages = []
#Opening the images
images = ["pallaPaint4soloCenterRight.png", "pallaPaint2ExternalNoise.png", "pallaPaint3LinearNoise.png" ,  "pallaPaint3soloTopLeft.png" , "pallaPaint4soloCenterRight.png", "pallaPaint1soloDownLeft.png",  "pallaPaint3ExternalNoise.png"  ,"pallaPaint3soloCenterCenter.png" , "pallaPaint3soloTopRight.png" , "randomNoise.png", "pallaPaint1soloTopCenter.png" , "pallaPaint3InternalNoise.png" , "pallaPaint3soloDownRight.png" ,"pallaPaint4soloCenterCenter.png", "linearNoise.png"]


def EM(originalImage, C = 0, rounds = 4, visual = 0):
        #originalImage := the image where to find the circle
        #C := the (optional) guess of the circle object
        #visual := run the function showing (1) or not showing the prints in each step of the algorithm

       
    edges = cv2.equalizeHist(originalImage)#histogram equalization
    
    #edge detection and threshold      
    edges = cv2.blur(edges, (7,7))
    edges = cv2.Canny(edges, threshold1=100, threshold2=200)



    #processing the image
    if len(edges.shape) == 3:#in that case we are analizing an RGB images
        image = cv2.cvtColor(edges, cv2.COLOR_GRB2GRAY) #converting the image to grayscale, non so se giusta conversione
    else:
        assert (len(edges.shape) == 2) # in that case we expect the image to be already in grayscale format
        image = edges
 
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > 200:
                image[i,j] = 255

    #setting the circle guess in case it is not defined
    if C == 0:#if we have no initial guess we start from the circle centered in the center of the image and with radious 1/3 of the smallest edge of the image
        C = circle(image.shape[0]/2,image.shape[1]/2, min(image.shape)/3)
    
    if visual:
        matplotlib.rcParams['figure.figsize'] = [15, 7]
    
        #printing the original image
        plt.subplot(121)
        plt.title('Original image.')
        plt.imshow(originalImage)
        
        #printing the processed image also displaying our guess
        plt.subplot(122)
        plt.title('Processed image with initial circle guess.')
        plt.imshow(C.onImage(image))
        plt.show()
        
    C.sigma = 30000
    
    for _ in range(rounds):
        
        #cycling in the image pixels in order to compute:
        #   the values delta_k for each pixel of the image (stored in dk_all)
        #   the values delta_k for each pixel of the image representing one (or 255) (stored in dk_1)
        #   the matrix M       
        M = []
        rk_quad_1 = []
        dk_1 = []
        dk_all = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                dk, rk_quad = ex.deltak(i,j,C)
                dk_all.append(dk)
                if image[i,j] == 255:
                    M.append([i**2+j**2,i,j,1])
                    dk_1.append(dk)
                    rk_quad_1.append(rk_quad)
                    
        #computations using the quantities computed above
        
        print("Sigma = {}.\n".format(C.sigma))
        p = ((len(M) - np.sum(scipy.stats.norm.pdf(dk_all, loc=0, scale = C.sigma)))/(image.shape[0]*image.shape[1])) 
        print(np.sum(scipy.stats.norm.pdf(dk_all, loc=0, scale = C.sigma)))
        print("P = {}.\n".format(p))       
        
        wk_1 = np.array([ex.wk(d, C.sigma, p) for d in dk_1])
        
        W = np.diag(wk_1)
        v = ma.computeEigenvector(M, W)
        C = ma.updateCircle(v)
        
        C.sigma = np.sum(np.array(rk_quad_1)*wk_1)/np.sum(wk_1)#update sigma
        
        if visual:
            #visualize the actual guess
            matplotlib.rcParams['figure.figsize'] = [7, 7]
            plt.title('Circle extimation after {} step.'.format(_+1))
            plt.imshow(C.onImage(image))
            plt.show()

    if visual:
        #representation of the extimated circle on the original imageName
        matplotlib.rcParams['figure.figsize'] = [7, 7]
        plt.title('Final extimation')
        plt.imshow(C.onImage(originalImage))
        plt.show()
        
        
        
image = cv2.imread("./../../Samples/DallE2/DallE2_0.png", 0)
        
EM(image, visual = 1)    

image = cv2.imread("./../../Samples/DallE2/DallE2_1.png", 0)
        
EM(image, visual = 1) 

image = cv2.imread("./../../Samples/DallE2/DallE2_2.png", 0)
        
EM(image, visual = 1) 
        
        
        
        
for imageName in images:
	print("Image: "+imageName+".\n\n")
	image = cv2.imread("./../../Samples/EM/"+imageName, 0)
	EM(image, visual = 1)

