import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from configuration import title

def main():
    global colors
    colors = {0 : 'blue', 1 : 'red', 2 : 'black', 3 : 'black', 4 : 'black',
     5 : 'black', 6 : 'black', 7 : 'black', 8 : 'black', 9 : 'black', 10 : 'black', 11 : 'black', 12 : 'black'}
    global F
    global index
    #user_input = input('Data filename:')
    #user_input = ('50,50,1000,0.2,25,0.55,10,4236274i_159.npy')
    F = np.load('StateData.npy')
    #index = int(user_input.split('i_')[1].split('.')[0])
    index = 0
    print(F.shape)
    fig, ax = plt.subplots()
    global lines
    '''F = []
    for i in range(1,16):
        F.append(list(array))
        array[i-1] = 0
        array[i] = 1'''
    width = int(title.split(',')[0])
    height = int(title.split(',')[1])
    #runtime = int(title.split(',')[2])
    global tot
    tot = width*height
    lines = []
    for i in range(tot):
        x = index_to_xy(i,width)[0]
        y = index_to_xy(i,width)[1]
        lines.append(ax.plot(x, y, color='green', marker = 'h', ls = '', markersize = 5.5)[0])
    anim = FuncAnimation(fig, animate, interval=100, frames=300)
    #print(user_input)
    #plt.title(user_input)
    #name = user_input.split('.')[0] + ".gif"
    plt.title(title)
    name = title + ".gif"
    anim.save(name)
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
    global tot
    global index
    '''j = index + i
    if j > (len(F)/tot - 1):
        j = j % 300
    print(j)
    print(len(F))'''
    x = F[i*tot:(i+1)*tot]
    for k,item in enumerate(lines):
        level = x[k]
        color = colors[level]
        item.set_color(color)


if __name__ == "__main__":
    main()