import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    colors = {0 : 'blue', 1 : 'red', 2 : 'black', 3 : 'dark grey', 4 : 'light grey'}
    array = np.zeros((16))
    array[0] = 1
    fig, ax = plt.subplots()
    global lines
    global F
    F = []
    for i in range(2,16):
        F.append(list(array))
        array[i-1] = 0
        array[i] = 1 
    lines = []
    for i in len(array):
        x = index_to_xy(i)[0]
        y = index_to_xy(i)[1]
        lines.append(ax.plot(x, y, color='green', marker = 'o', ls = '', markersize = 25)[0])
    anim = FuncAnimation(fig, animate, interval=100, frames=15)
    anim.save('Animation.gif', writer='Dan')
    plt.draw()
    plt.show()

def index_to_xy(index, width):
    row = int(str(index / width)[0])
    y = row
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * width)
    else:
        x = index - (row * width) + 0.5
    return (x,y)

def animate(i):
    global lines
    global F
    global colors
    x = F[i]
    for j,item in enumerate(lines):
        level = x[j]
        color = colors[level]
        lines.set_color(color)


if __name__ == "__main__":
    main()