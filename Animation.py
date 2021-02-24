import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as mpa
from matplotlib.animation import FuncAnimation
from hexalattice.hexalattice import *
from matplotlib.lines import Line2D
import time

#For each run - there will be some boolean value that says whether to animate
#If this is true - column fname will contain the filename for that run.
#Use this filename to access the correct .npy file and animate accordingly.

#Needs to be generalised for a full run and for a transition.

def Animate(fname, type_, loc2, loc3, loc4,adds, loc_off = False, ham_off = False):
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
    if type_ == 'transition' or 'full': #Setting the runtime
        runtime = len(F)//(tot)  
        print(runtime)                                                         
    else:
        pass

    lines = []
    for i in range(tot):
        x,y = index_to_xy(i,width) #Getting the real space coordinates.
        lines.append(ax.plot(x, y, color='green', marker = 'h', ls = '', markersize = 40)[0]) #markersize = 5.5 for 100x100
        pass


    if loc_off:
        anim = FuncAnimation(fig, animate_func, interval=3000, frames=runtime, fargs = (F, lines, colors, tot, ham_off))
    else:
        anim = FuncAnimation(fig, animate_func_loc, interval=100, frames=runtime, fargs = (F, lines, colors, tot, loc2, loc3, loc4))

    #Fix to put A,a,o in animation title
    A,a,o = str(adds).split('[')[1].split(']')[0].split(',')
    print(A)
    if '.' in a:
        amp = str(a.split('.')[0]) + str(a.split('.')[1])
    else:
        amp = a
    if '.' in o:
        off = str(o.split('.')[0]) + str(o.split('.')[1])
    else:
        off = o
    if ham_off:
        plt.title(fname + "\n, A" + str(A) + ", amp" + str(amp) + ", offset" + str(off))
    else:
        plt.title('Toy Animation to understand Hamming Distance')
        fname = 'ToyAnim'
    name = fname + ", A" + str(A) + ", amp" + str(amp) + ", offset" + str(off) +".gif"

    anim.save(name, writer = mpa.PillowWriter(fps=1))
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

def ham_dis_anim(time_data):
    activated_sites = np.where(time_data == 1)[0]
    activated_sites_x = [index_to_xy(i,8)[0] for i in activated_sites]
    if len(activated_sites) > 0:
        x_mean = np.mean(activated_sites_x)
        Ham_dis = np.sum((activated_sites_x-x_mean)**2)/len(activated_sites_x)
        return np.sqrt(Ham_dis)
    else:
        return 0.00


def animate_func(i, F, lines, colors, tot, ham_off):
    x = F[i*tot:(i+1)*tot] #data for each time step.
    for k,item in enumerate(lines):
        level = x[k]
        color = colors[level]
        item.set_color(color)
    if not ham_off:
        ham = round(ham_dis_anim(x),2)
        label_ham = 'Hamming Distance = ' + str(ham)   
        legend_elements = [Line2D([0], [0], marker='o', color='white', label=label_ham, markerfacecolor='white', markersize=0)]
        plt.legend(handles = legend_elements, bbox_to_anchor=(0.75, -0.05), fontsize = 12)

def animate_func_loc(i, F, lines, colors, tot, loc2, loc3, loc4):
    x = F[i*tot:(i+1)*tot] #data for each time step.
    for k,item in enumerate(lines):
        level = x[k]
        color = colors[level]
        item.set_color(color)
    lines[loc2].set_color('green')
    lines[loc3].set_color('yellow')
    lines[loc4].set_color('orange')
