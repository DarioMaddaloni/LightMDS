import cv2
from eraser import eraser

screen="Drawing"
img=cv2.imread("image.png")
cv2.namedWindow(screen)

eraserObj = eraser(screen, img)
cv2.setMouseCallback(screen, eraserObj.handleMouseEvent)
# show initial image
cv2.imshow(screen, img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("iamge", img)
cv2.waitKey(0)
