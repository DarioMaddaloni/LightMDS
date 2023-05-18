import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class eraser:
    def __init__(self, image, alwaysRefresh = 0):

        # Defining plot objects
        self.fig, self.ax = plt.subplots()
        self.imsh = None
        self.image = image

        # Circle (eraser appearance)
        self.circle = matplotlib.patches.Circle(xy=(512, 512), radius=30, color='yellow',
                                           fill=False, alpha=0.4)
        self.ax.add_patch(self.circle)

        # Mouse event variables
        self.press = False
        self.position = None

        # Speed up or not
        self.alwaysRefresh = alwaysRefresh

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.circle.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def show(self):
        """Intruducing the slider for the radius and showing the interface."""
        self.imsh = plt.imshow(self.image, cmap='gray', vmin=0, vmax=255)

        # Make a vertically oriented slider to control the radius
        axSlider = self.fig.add_axes([0.1, 0.15, 0.0225, 0.63])
        radiusSlider = Slider(
            ax=axSlider,
            label="Radius width",
            valmin=10,
            valmax=150,
            valinit=40,
            orientation="vertical"
        )

        # The function to be called anytime a slider's value changes
        def update(val):
            self.circle.set_radius(val) # Update radius value
            self.circle.figure.canvas.draw() # Refresh the image

        # Register the update function
        radiusSlider.on_changed(update)

        # Showing the interface
        self.fig.suptitle('Please erase noise and then close this window.', fontsize=16)
        plt.show()

    def on_press(self, event):
        """Start erasing at mouse click"""

        if event.inaxes != self.circle.axes or not(event.inaxes == self.ax):
            return

        # Update press variable
        self.press = True

        # Update the position of the event
        self.position = (event.xdata, event.ydata)

        # Move and fill the circle
        self.circle.set(fill= True, center= (event.xdata, event.ydata))
        self.circle.figure.canvas.draw()

        # Drawing black pixels on the image cycling through the x and y coordinates of the points of the ball
        for i in range(int( max(0,self.position[0]-self.circle.get_radius())), int(min(self.image.shape[0],self.position[0]+self.circle.get_radius()))):
            value = np.floor(np.sqrt(max(0,self.circle.get_radius()**2 - (i-self.position[0]-1)**2))) # Pitagoras triangle
            m = int(max(0,self.position[1]-value)) # x inf of the ball for coordinate i
            M = int(min(self.image.shape[0],self.position[1]+value)) # x sup of the ball for coordinate i
            self.image[m : M,i] = 0 # Adding the 0 pixels

            # Refreshing the image
            if self.alwaysRefresh:
                self.imsh.set_data(self.image)
                plt.draw()

    def on_motion(self, event):
        """Update the rubber appearance during motion and continue erasing if the mouse button is pressed"""
        if self.position is None or event.inaxes != self.circle.axes or not(event.inaxes == self.ax):
            return

        # Update the position of the event
        self.position = (event.xdata, event.ydata)
        self.circle.set(center= self.position)
        self.circle.figure.canvas.draw()

        # Continue erasing
        if self.press:
            """Same code of on_press"""

            # Drawing black pixels on the image cycling through the x and y coordinates of the points of the ball
            for i in range(int(max(0, self.position[0] - self.circle.get_radius())),
                           int(min(self.image.shape[0], self.position[0] + self.circle.get_radius()))):
                value = np.floor(np.sqrt(
                    max(0, self.circle.get_radius() ** 2 - (i - self.position[0] - 1) ** 2)))  # Pitagoras triangle
                m = int(max(0, self.position[1] - value))  # x inf of the ball for coordinate i
                M = int(min(self.image.shape[0], self.position[1] + value))  # x sup of the ball for coordinate i
                self.image[m: M, i] = 0  # Adding the 0 pixels

            # Refreshing the image
            if self.alwaysRefresh:
                self.imsh.set_data(self.image)
                plt.draw()

    def on_release(self, event):
        """Return to initial variable configuration"""
        self.press = False
        self.circle.set(fill= False)
        self.circle.figure.canvas.draw()

        # Refreshing the image in the case this has never happened before (fast mode)
        if not self.alwaysRefresh:
            self.imsh.set_data(self.image)
            plt.draw()

    def close(self):
        """Disconnect all callbacks and return the resulting image."""
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
        return self.image

# Test
if __name__ == '__main__':

    # Create a test image
    image = np.zeros((1024,1024))
    image[300:500, 200:500] = 255
    image[40:700, 900:1000] = 255

    # Call eraser methods
    e = eraser( image, alwaysRefresh=0)
    e.connect()
    e.show()

    # Obtain the resulting image
    img = e.close()

    # Show the result
    #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    #plt.show()




