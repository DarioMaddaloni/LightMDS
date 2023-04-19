###	 EXPECTATION MAXIMIZATION MAIN FILE

import sys

sys.path.insert(0, './../..')
from Code.import_file import *
from Code.circle_lib import circle
from Code.eraser_lib import eraser
from Code.show_lib import *
import expectation_lib as ex
import maximization_lib as ma
from interaction_lib import guess3, guess3evo

# definition of sharpening used in preprocessing
from scipy.ndimage import gaussian_filter


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    sharpened = img + alpha * (img - filter_blurred_f)
    return sharpened


def EM_evo(originalImage, C=0, rounds=4, visual=0, visualFinal=1, erase=1):
    """
		originalImage := the image where to find the circle
		C := the (optional) guess of the circle object
		visual := run the function showing (1) or not showing the prints in each step of the algorithm
	"""

    print("From RGB to GRAY")  # processing the image
    if len(originalImage.shape) == 3:  # in that case we are analizing an RGB images
        image = cv2.cvtColor(originalImage, cv2.COLOR_RGB2GRAY)  # converting the image to grayscale
    else:
        assert (len(originalImage.shape) == 2)  # in that case we expect the image to be already in grayscale format
        image = originalImage

    print("sharpening")
    image = sharpening(image, 0.12, 3)

    # edge detection and threshold
    print("blurring")
    image = cv2.blur(image, (11, 11))
    # Try sharpening instead of blurring and anjust the thresholds of cannying
    print("Cannying")  # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    image = cv2.Canny(image, threshold1=30, threshold2=70)  # 60, 100

    print("black or white")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 125:
                image[i, j] = 255
            else:
                image[i, j] = 0

    if erase:  # Erase noise by hand
        screen = "Figure 1"
        cv2.namedWindow(screen, cv2.WINDOW_NORMAL)
        eraserObj = eraser(screen, image, radius=30)
        cv2.setMouseCallback(screen, eraserObj.handleMouseEvent)
        # show initial image
        cv2.imshow(screen, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # setting the circle guess in case it is not defined
    if C == 0:  # if we have no initial guess we start from the circle centered in the center of the image and with radious 1/3 of the smallest edge of the image
        C = circle(image.shape[0] / 2, image.shape[1] / 2, min(image.shape) / 3)
        C.sigma = C.r * 200
    else:
        C.sigma = 80
    # the value sigma differs in the case of accurate initial guess or random guess

    print("Show images")
    if visual:
        matplotlib.rcParams['figure.figsize'] = [15, 15]

        # printing the processed image also displaying our guess
        plt.title('Processed image with initial circle guess.')
        plt.imshow(C.onImage(image))
        plt.show()

    print("raggio = {}".format(C.r))

    epsilon = 0.2

    print("epsilon = {}.\n".format(epsilon))

    for _ in range(rounds):
        print("Execute round {}/{} with parameters:\n".format(_ + 1, rounds))
        # cycling in the image pixels in order to compute:
        #	 the values delta_k for each pixel of the image representing one (or 255) (stored in dk_1)
        #	 the matrix M
        M = []
        dk_1 = []

        if visual:
            ex.plot_prob_curve_evo(C.sigma, epsilon)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                if (image[i, j] == 255 and (dk := ex.deltak_evo(i, j,
                                                                C)) < 3000):  # la seconda condizione è messa per dare un significativo speedup
                    # dk = ex.deltak_evo(i,j,C)
                    M.append([i ** 2 + j ** 2, i, j, 1])
                    dk_1.append(dk)

        print("Sigma = {}.\n".format(C.sigma))

        wk_1 = np.array([ex.wk_evo(d, C.sigma, epsilon) for d in dk_1])
        wk_2 = np.array([ex.wk_pro(d, C.sigma, epsilon) for d in dk_1])

        W = np.diag(wk_1)

        v = ma.computeEigenvector(M, W)

        C = ma.updateCircle(v)

        C.sigma = ex.updateSigma(wk_2, dk_1)  # np.sum(np.array(dk_1)*wk_1)/np.sum(wk_1)#update sigma

        print(C.cx)

        if visual:
            # visualize the actual guess
            matplotlib.rcParams['figure.figsize'] = [7, 7]
            plt.title('Circle estimation after {} step.'.format(_ + 1))
            plt.imshow(C.onImage(image))
            plt.show()

    if visualFinal:
        # visualize the actual guess
        matplotlib.rcParams['figure.figsize'] = [7, 7]
        plt.title('Final estimation')
        plt.imshow(C.onImage(image))
        plt.show()


for i in range(1, 10):  # loop in the DallE2-generated database
    print("Image {}".format(i) + ".\n")
    image = cv2.imread("./../../Samples/DallE2/DallE2_{}.png".format(i), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    C = guess3evo(image)
    rounds = 10

    EM_evo(image, C, rounds, visual=0, visualFinal=1, erase=0)
