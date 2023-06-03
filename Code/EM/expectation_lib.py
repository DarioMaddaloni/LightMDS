from Code.import_file import *
from Code.show_lib import *
from Code.LightAnalisys.geometry_lib import *


def deltak_evo(xk, yk, circleObj): #not returning the rk_quand since no more needed
	cx, cy, r = circleObj.center.x, circleObj.center.y, circleObj.r
	return np.abs( (xk-cx)**2 + (yk - cy)**2 - r**2 )


def wk(dk, sigma, epsilon):
	value = np.exp( - (dk) / (2 * (sigma ** 2)))
	return (value) / (value + epsilon)


def wk_sigma(dk, sigma, epsilon):
	value = np.exp( - (dk**2) / (2 * (sigma ** 2)))
	return (value) / (value + epsilon)


def plot_prob_curve_evo(sigma, p):
    x = np.arange(0, 50, 1)
    graph = wk_sigma(x ** 2, sigma, p)
    plt.title("Sigma = {}, Epsilon = {}".format(sigma,p))
    plt.ylim([0,1]) # setting the y interval to the unitary one
    plt.plot(x, graph)
    plt.show()
