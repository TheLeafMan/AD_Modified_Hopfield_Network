import numpy as np
import matplotlib.pyplot as plt

def toggle_neurons(grid_size):
    # Initialize the grid with all neurons set to 1 (all of them are black  initally)
    grid = np.ones((grid_size, grid_size))

    # Function to handle click events
    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            # centers the click radius 

            grid[y , x] = -grid[y, x]  # Toggle neuron state (black to white or white to black)
            update_plot()

    # Function to update the plot
    def update_plot():
        ax.clear()
        # shows the opposite of the grid, so that black is 1 and white is -1
        ax.imshow(grid, cmap='grey')
        plt.draw()

    # visulize the grid 

    fig, ax = plt.subplots()
    update_plot()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    print(grid)

# Example usage
toggle_neurons(16)