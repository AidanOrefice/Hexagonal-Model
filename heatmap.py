import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Hexagon import HexagonalLattice
from hexalattice.hexalattice import *
from CouplingViz import sinusoid2D


#Need to think about how it is normalised.

def value_counts(df,n):
    #n relates to which location method i.e. location_n
    locs = df['location_' + str(n)].value_counts(sort = False, normalize = True)
    
    inds = list(map(int, list(locs.index)))
    locs_ = pd.DataFrame(np.zeros(10000))
    for i in range(10000):
        if i in inds:
            locs_[0][i] = locs[str(i)]

    return np.array(locs_[0])

def plot_heat_map(fname):
    runs = pd.read_csv(fname + '.csv')
    #print(type(runs.iloc[6438,21]))
    #Fix problem of trues in location 4, set to location 3 value
    runs.loc[runs['location_4'] == 'True', 'location_4'] = runs.loc[runs['location_4'] == 'True']['location_3']
    runs = runs.loc[runs['in AF?']] #Only look at ones that enter AF.

    #Amplitude above 0.5
    #Amplitude below 0.1
    #LIMITS
    for ind, row in runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        amp, offs = float(list_[1]), float(list_[2])
        if (amp > 0.5) or (amp < 0.05) or (offs < 0.25):
            runs = runs.drop(index = ind)
    
    A = int(fname.split('_')[-1]) #Need to pull out the value of A from fname, dont remember the format.

    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(16,7)
    hex_centers, ax1 = create_hex_grid(nx=100, ny=100, do_plot=True, align_to_origin = False, h_ax = ax1)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]

    loc2 = value_counts(runs,2)
    loc3 = value_counts(runs,3)
    loc4 = value_counts(runs,4)

    locs = (loc2+loc3+loc4)/3


    loc_plot = ax1.scatter(x,y,marker = 'h', s=17, c = locs, cmap = 'gnuplot2') # gist_gray, gnuplot
    fig.colorbar(loc_plot, ax = ax1, shrink = 0.6)

    #Plot the sapce onto the same figure.
    _, ax2 = create_hex_grid(nx=100,ny=100, do_plot=True, align_to_origin = False, h_ax = ax2)

    #0.1, 1 - amp,offs: I jsut chose arbitrary values but have to be careful with colorbars here.

    sin_z = [sinusoid2D(x[i], y[i], A, 0.1, 1) for i in range(len(x))]
    couple_plot = ax2.scatter(x,y,marker = 'h', s=17, c = sin_z)
    cbar = fig.colorbar(couple_plot, ax = ax2, shrink = 0.6, ticks = [min(sin_z), max(sin_z)])

    cbar.ax.set_yticklabels(['LOW', 'HIGH'])  # vertically oriented colorbar
    
    
    fig.suptitle('Heatmaps of location of AF induction and the Corresponding Coupling Space', fontsize = 16)
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.close()

    
plot_heat_map('Normal_Modes_Phase_Space_5')