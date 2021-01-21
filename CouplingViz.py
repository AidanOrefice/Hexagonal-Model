import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from hexalattice.hexalattice import *
from configuration import config


"""
Script that visualises the coupling in the lattice.

Uses https://github.com/alexkaz2/hexalattice/blob/master/hexalattice/hexalattice.py
"""
amp = 0.2
mean = 0.75

Ax = 5
Ay = 1

rows = 80
columns = 140 

def sinusoid2D(x, y, Ax=Ax, Ay=Ay,  amp = amp, mean = mean):
    #Amplitude - directly sets the amplitude of the function
    #Mean - directly sets the offset/mean of the function.
    # 0 < (Mean +/- amp) < 1 
    #Ax/Ay stretch out the modes.
    #Ay must be an integer value to ensure periodicity.
    return (amp/2)*(np.sin(Ax*x*(2*np.pi/columns))+np.sin(Ay*y*(2*np.pi/(rows  - rows*(1-(np.sqrt(3)/2)))))) + mean

def gradient(x,start=0.8,end = 0.6):
    delta = (end-start)/50
    return (delta*x) + start

def index_to_xy(index):
    row = np.floor(index / 50)
    y = row  - row*(1-(np.sqrt(3)/2))#fix.
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * 50)
    else:
        x = index - (row * 50) + 0.5
    return x,y

def VizTest(Ax,Ay,amp,mean,rows,columns):
    fig,ax = plt.subplots()
    fig.set_size_inches(16,9)
    hex_centers, ax = create_hex_grid(nx=columns,ny=rows, do_plot=True, align_to_origin = False, h_ax = ax)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers] 

    sin_z = [sinusoid2D(x[i], y[i]) for i in range(len(x))]
    grad_z = [gradient(i) for i in x]
    a = ax.scatter(x,y,marker = 'h', s=17, c = sin_z)
    fig.colorbar(a,shrink=0.8)

    #print(np.mean(sin_z))
    #print(np.var(sin_z))
    #print(np.std(sin_z))

    label_mean = 'Offset = ' + str(mean)
    label_amp = 'Amplitude = ' + str(amp)

    legend_elements = [Line2D([0], [0], marker='o', color='white', label=label_mean, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_amp, markerfacecolor='white', markersize=0)]



    plt.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize = 20)

    plt.title(r"$\frac{Amplitude}{2} \times \left( \sin(%.3f*\frac{2\pi x}{length}) + \sin(%.3f*\frac{2\pi y}{height}) \right) + Offset$" %(Ax,Ay), fontsize = 20)
    title = 'CouplingViz_%i,%i,%i,%i,%i,%i.png' %(amp*100,mean*100,Ax,Ay,rows,columns)
    plt.savefig(title)
    plt.close()

VizTest(Ax,Ay,amp,mean,rows,columns)