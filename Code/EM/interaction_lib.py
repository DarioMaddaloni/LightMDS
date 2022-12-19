import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2
import math
from Code.circle_lib import circle




image = cv2.imread("./../../Samples/DallE2/DallE2_{}.png".format(0), 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)



def tellme(s):
    plt.title(s, fontsize=16)
    plt.draw()


def guess(originalImage):
    
    image = np.zeros_like(originalImage)
    #for i in range(originalImage.shape[0]):
    #    image[i] = originalImage[originalImage.shape[0]-1-i]  
    image = originalImage      
    plt.imshow(image)
    #plt.show()

    tellme('You will define a circle, click to begin')

    plt.waitforbuttonpress()

    while True:
        center = []
        while len(center) < 1:
            tellme('Select the center with mouse')
            center = np.asarray(plt.ginput(1, timeout=-1))
            if len(center) < 1:
                tellme('Too few points, starting over')
                time.sleep(1)  # Wait a second
                
        pts = []
        while len(pts) < 1:
            tellme('Select a point on the border')
            pts = np.asarray(plt.ginput(1, timeout=-1))
            print(pts)
            if len(pts) < 1:
                tellme('Too few points, starting over')
        
        r = math.dist(pts[0],center[0])
        
        C = matplotlib.patches.Circle((center[0][0],center[0][1]), radius=r, color='r',alpha = 0.4)
        
        
        fig = plt.gcf()
        ax = fig.gca()
        
        ax.add_patch(C)

        tellme('Happy? Key click for yes, mouse click for no')

        if plt.waitforbuttonpress():
            break
            
        C.remove()
    return circle(center[0][1],center[0][0],r)
    
    
