import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def main():
    global colors
    colors = {0 : 'blue', 1 : 'red', 2 : 'black', 3 : 'dimgrey', 4 : 'lightgrey'}
    global F
    F = np.load('StateData.npy')
    print(F.shape)
    fig, ax = plt.subplots()
    global lines
    '''F = []
    for i in range(1,16):
        F.append(list(array))
        array[i-1] = 0
        array[i] = 1'''
    f = open('settings.txt', 'r')
    settings = f.read()
    width = int(str(settings).split(',')[0])
    lines = []
    for i in range(F.shape[1]):
        x = index_to_xy(i,width)[0]
        y = index_to_xy(i,width)[1]
        lines.append(ax.plot(x, y, color='green', marker = 'h', ls = '', markersize = 1)[0])
    anim = FuncAnimation(fig, animate, interval=100, frames=F.shape[0])
    print(settings)
    plt.title(settings)
    name = str(settings) + ".gif"
    anim.save(name, writer='Dan')
    plt.draw()
    plt.show()

def index_to_xy(index, width):
    row = math.floor(index / width)
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
        item.set_color(color)


if __name__ == "__main__":
    main()