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

A = 3

rows = 100
columns = 100 

def index_to_xy(index):
    row = np.floor(index / 100)
    y = row - row*(1-(np.sqrt(3)/2)) #fix.
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * 100)
    else:
        x = index - (row * 100) + 0.5
    return (x,y)

def sinusoid2D(x, y, A = A,  amp = amp, mean = mean):
    #Amplitude - directly sets the amplitude of the function
    #Mean - directly sets the offset/mean of the function.
    # 0 < (Mean +/- amp) < 1 
    #A stretch out the modes. A must be an integer value.
    return (amp/2)*(np.cos(A*x*(2*np.pi/columns))+np.cos(A*y*(2*np.pi/(rows  - rows*(1-(np.sqrt(3)/2)))))) + mean #(x+25), (y+20) - ideal unit cell

def gradient(x,start=0.8,end = 0.6):
    delta = (end-start)/50
    return (delta*x) + start

def VizTest(A,amp,offs,rows,columns):
    fig,ax = plt.subplots()
    fig.set_size_inches(16,9)
    hex_centers, ax = create_hex_grid(nx=columns,ny=rows, do_plot=True, align_to_origin = False, h_ax = ax)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers] 

    sin_z = [sinusoid2D(x[i], y[i], A, amp, offs) for i in range(len(x))]
    a = ax.scatter(x,y,marker = 'h', s=17, c = sin_z) #s=17
    fig.colorbar(a,shrink=0.8)


    label_offs = 'Offset = ' + str(offs)
    label_amp = 'Amplitude = ' + str(amp)
    label_a = r'$A$ = ' + str(A)
    label_mean = 'Mean = ' + str(round(np.mean(sin_z),3))
    label_variance = 'Variance = ' + str(round(np.var(sin_z),3))    

    legend_elements = [Line2D([0], [0], marker='o', color='white', label=label_offs, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_amp, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_a, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_mean, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_variance, markerfacecolor='white', markersize=0)]



    plt.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.025), ncol=3, fontsize = 14)

    plt.title(r"$\frac{Amplitude}{2} \times \left( \sin(A\frac{2\pi x}{length}) + \sin(A\frac{2\pi y}{height}) \right) + Offset$", fontsize = 20)
    title = 'CouplingViz_%i,%i,%i,%i,%i.png' %(amp*100,mean*100,A,rows,columns)
    plt.savefig(title)
    plt.close()

    for ind,val in enumerate(sin_z):
        if val >  0.945:
            x,y = index_to_xy(ind)
            if (x == 8.5) or (x==41.5) or (x==74.5):
                print(x,y)
                print(ind)


if __name__ == '__main__':
    t0 = time.time()
    VizTest(A,amp,mean,rows,columns)
    t1 = time.time()
    print(t1-t0)

