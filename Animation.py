import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

#For each run - there will be some boolean value that says whether to animate
#If this is true - column fname will contain the filename for that run.
#Use this filename to access the correct .npy file and animate accordingly.

#Needs to be generalised for a full run and for a transition.

def Animate(fname, type, loc2, loc3, loc4,adds):
    #data_file - filename of the .npy file 
    #type - type of animation to run e.g. full or transition.

    colors = {0 : 'blue', 1 : 'red', 2 : 'black', 3 : 'black', 4 : 'black',
     5 : 'black', 6 : 'black', 7 : 'black', 8 : 'black', 9 : 'black', 10 : 'black', 11 : 'black', 12 : 'black'}

    data_file = fname + '.npy' #Loading in data
    F = np.load(data_file)
    print(F.shape)
    fig, ax = plt.subplots()
    
    width = int(data_file.split(',')[0]) # First three inputs
    height = int(data_file.split(',')[1])
    runtime = int(data_file.split(',')[2])
    

    tot = width*height #Length of 1D array 
    if type == 'transition': #Setting the runtime
        runtime = len(F)//(tot)  
        print(runtime)                                                         
    else:
        pass

    lines = []
    for i in range(tot):
        x,y = index_to_xy(i,width) #Getting the real space coordinates.
        lines.append(ax.plot(x, y, color='green', marker = 'h', ls = '', markersize = 5.5)[0])
    anim = FuncAnimation(fig, animate_func_loc, interval=100, frames=runtime, fargs = (F, lines, colors, tot, loc2, loc3, loc4))

    #Fix to put A,a,o in animation title

    A,a,o = str(adds).split('[')[1].split(']')[0].split(',')
    if '.' in a:
        amp = str(a.split('.')[0]) + str(a.split('.')[1])
    else:
        amp = a
    if '.' in o:
        off = str(o.split('.')[0]) + str(o.split('.')[1])
    else:
        off = o
    plt.title(fname + "\n, A" + str(A) + ", amp" + str(amp) + ", offset" + str(off))
    #Careful to avoid fullstop in filename
    name = fname + ", A" + str(A) + ", amp" + str(amp) + ", offset" + str(off) +".gif"
    anim.save(name)
    plt.close()

def index_to_xy(index, width):
    row = np.floor(index / width)
    y = row - row * (1 - (np.sqrt(3) / 2)) #fix.
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * width)
    else:
        x = index - (row * width) + 0.5
    return (x,y)


def animate_func(i, F, lines, colors, tot):
    x = F[i*tot:(i+1)*tot] #data for each time step.
    for k,item in enumerate(lines):
        level = x[k]
        color = colors[level]
        item.set_color(color)

def animate_func_loc(i, F, lines, colors, tot, loc2, loc3, loc4):
    x = F[i*tot:(i+1)*tot] #data for each time step.
    for k,item in enumerate(lines):
        level = x[k]
        color = colors[level]
        item.set_color(color)
    lines[loc2].set_color('green')
    lines[loc3].set_color('yellow')
    lines[loc4].set_color('orange')
