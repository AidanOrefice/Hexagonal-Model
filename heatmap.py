import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Hexagon import HexagonalLattice
from CouplingViz import sinusoid2D


#This is so rough its unreal, I havent even tested it yet

def plot_heat_map(fname):
    runs = pd.read_csv(fname + '.csv')

    A = fname[-1] #Need to pull out the value of A from fname, dont remember the format.

    fig,(ax1,ax2) = plt.subplots(2,1)
    hex_centers, ax1 = create_hex_grid(nx=100, ny=100, do_plot=True, align_to_origin = False, h_ax = ax1)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]

    #Need to maybe do checks for locations to evaluate whether each location is valid
    loc2 = np.array([runs['location_2'].count(i) for i in range(100*100 -1)])
    loc3 = np.array([runs['location_3'].count(i) for i in range(100*100 -1)])
    loc4 = np.array([runs['location_4'].count(i) for i in range(100*100 -1)])
    locs = (loc2 + loc3 + loc4)/(3*len(runs)) #Normalising

    ax1.scatter(x,y,marker = 'h', s=17, c = locs)

    #Plot the sapce onto the same figure.
    _, ax2 = create_hex_grid(nx=100,ny=100, do_plot=True, align_to_origin = False, h_ax = ax2)

    #0.1, 1 - amp,offs: I jsut chose arbitrary values but have to be careful with colorbars here.
    sin_z = [sinusoid2D(x[i], y[i], A, 0.1, 1) for i in range(len(x))]
    a = ax2.scatter(x,y,marker = 'h', s=17, c = sin_z)
    #fig.colorbar(a,shrink=0.8)

    plt.title('Heatmaps of location of AF induction And the Coupling Space')
    plt.savefig('heatmap.png')
    plt.close()

    

