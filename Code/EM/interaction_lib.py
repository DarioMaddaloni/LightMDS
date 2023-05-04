import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2
import math
from Code.circle_lib import circle


def tellme(s):
	plt.title(s, fontsize=16)
	plt.draw()
	
	
def interactiveGuess(image, max_y_difference = 20):#guessing 3 points in the border of the circle. max_y_difference is the maximal acceptable difference between two y coordinates in order to find the straigt lines intersection

    print("\n --- START OF INTERACTIVE GUESS --- ")

    plt.imshow(image)
    
    fig = plt.gcf()
    ax = fig.gca()

    tellme('You will define a circle, click to begin.')

    plt.waitforbuttonpress()

    while True:

        pts = []
        while len(pts) < 3 or len(intersection)<1:

            print("\nSelecting three points on the border...")
            tellme('Please, select three points on the border.')
            pts = np.asarray(plt.ginput(3, timeout=-1))# Taking the points as input
            print("The selected points are: ", pts[0], pts[1], pts[2],".")
            
            print("Finding the intersection...")
            intersection = []# Here we will store the intersections between the two lines defined by couples of points (pts[0], pts[1]) and (pts[1], pts[2])
            if (abs(pts[1][1]-pts[0][1]) > 1/100) and (abs(pts[2][1]-pts[1][1]) > 1/100):    # We avoid the case (pts[0], pts[1]) and (pts[1], pts[2]) are vertical. This should happen with probability 0

                for x in range(image.shape[0]):# Cycling through the x coordinate of the image
                    
                    # y point coordinate in the first line
                    y_1 = -(pts[1][0]-pts[0][0])/(pts[1][1]-pts[0][1])*(x-(pts[1][0]+pts[0][0])/2)+(pts[1][1]+pts[0][1])/2
                    
                    # y point coordinate in the second line
                    y_2 = -(pts[2][0]-pts[1][0])/(pts[2][1]-pts[1][1])*(x-(pts[2][0]+pts[1][0])/2)+(pts[2][1]+pts[1][1])/2
                    
                    if abs(y_1-y_2) < max_y_difference:# Selecting the points if they are close to each other (less than max_y_difference)
                        intersection.append([x,y_1])
                        intersection.append([x,y_2])                
                    
            print("The intersection is: ", intersection)
            if len(intersection)<1:
                print("Empty intersection found.")
                tellme('Not able to find points in the intersection. Restarting...')
                pts = []
                time.sleep(1)  # Wait a second
                continue
                
            if (len(pts) < 3):
                print("Selected points problem.")
                tellme('Not enough points selected. Restarting...')
                pts = []
                time.sleep(1)  # Wait a second
            
        print("\nComputing the circle coordinates and radius...")
        center = np.mean(intersection, axis = 0)      # Averaging the coordinate values found in the intersection in order to discover the center
        
        r = math.dist(pts[0],(center[0], center[1]))    # Computing our guess for the radius
        
        # plotting the sphere on the image
        print("Showing the guess...")
        C = matplotlib.patches.Circle(center, radius=r, color='r',alpha = 0.4)      
        
        ax.add_patch(C)

        # The user can attempt to refine its guess
        tellme('Happy? Key click for yes, mouse click for no')

        if plt.waitforbuttonpress():
            plt.close()
            break
            
        C.remove()

    print("Circle returned.\n --- END OF INTERACTIVE GUESS --- ")

    return circle(center[1],center[0],r)# inverted coordinates for not well understood reason, but properly working
