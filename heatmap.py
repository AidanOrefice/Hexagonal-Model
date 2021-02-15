import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Hexagon import HexagonalLattice
from hexalattice.hexalattice import *
from CouplingViz import sinusoid2D
from scipy.spatial.distance import euclidean
import time


def gaussian(dist,c,theta):
    #List of distances to a points
    #List of c to that given point
    exponent = -0.5*(np.square(dist/theta) + np.square(c/theta))
    return np.exp(exponent)

def xy_to_index(x,y,width):
    if x % 1 == 0:
        row = int(x)
        col = int(y/(np.sqrt(3)/2))
    else:
        row = int(x-0.5)
        col = int(y/(np.sqrt(3)/2)) + 1
    idx = row + col*width
    return idx

def index_to_xy(index):
    row = np.floor(index / 100)
    y = row - row*(1-(np.sqrt(3)/2)) #fix.
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * 100)
    else:
        x = index - (row * 100) + 0.5
    return (x,y)

def value_counts(df,n):
    #n relates to which location method i.e. location_n
    locs = df['location_' + str(n)].value_counts(sort = False, normalize = False)
    
    inds = list(map(int, list(locs.index)))
    locs_ = pd.DataFrame(np.zeros(10000))
    for i in range(10000):
        if i in inds:
            locs_[0][i] = locs[str(i)] #str(i) or i - if key error occurs

    return np.array(locs_[0])

def plot_heat_map(fname, convolve = True):
    runs = pd.read_csv(fname + '.csv')
    #print(type(runs.iloc[6438,21]))
    #Fix problem of trues in location 4, set to location 3 value
    runs.loc[runs['location_4'] == 'True', 'location_4'] = runs.loc[runs['location_4'] == 'True']['location_3']
    runs = runs.loc[runs['in AF?']] #Only look at ones that enter AF.

    #Amplitude above 0.5
    #Amplitude below 0.1
    #LIMITS
    '''for ind, row in runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        amp, offs = float(list_[1]), float(list_[2])
        if (amp > 0.5) or (amp < 0.05) or (offs < 0.59):
            runs = runs.drop(index = ind)'''
    
    A = int(fname.split('_')[-1]) #Need to pull out the value of A from fname, dont remember the format.

    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(16,7)
    hex_centers, ax1 = create_hex_grid(nx=100, ny=100, do_plot=True, align_to_origin = False, h_ax = ax1)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]

    loc2 = value_counts(runs,2)
    loc3 = value_counts(runs,3)
    loc4 = value_counts(runs,4)

    locs = (loc2+loc3+loc4)/(3*len(runs))

    if convolve:
        locs = Convolve(locs, 3, 0.75)
        np.save('con_locs_{}.npy'.format(A), np.asanyarray(locs))
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
    name = 'heatmap_' + str(A) +'.png'
    if convolve:
        name = 'convolved_heatmap_' + str(A) +'.png'
    plt.savefig(name)
    plt.close()

def plot_heat_map_contour(fname, convolve = True):
    runs = pd.read_csv(fname + '.csv')
    #print(type(runs.iloc[6438,21]))
    #Fix problem of trues in location 4, set to location 3 value
    runs.loc[runs['location_4'] == 'True', 'location_4'] = runs.loc[runs['location_4'] == 'True']['location_3']
    runs = runs.loc[runs['in AF?']] #Only look at ones that enter AF.

    #Amplitude above 0.5
    #Amplitude below 0.1
    #LIMITS
    '''for ind, row in runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        amp, offs = float(list_[1]), float(list_[2])
        if (amp > 0.5) or (amp < 0.05) or (offs < 0.59):
            runs = runs.drop(index = ind)'''
    
    A = int(fname.split('_')[-1]) #Need to pull out the value of A from fname, dont remember the format.

    fig,ax1 = plt.subplots()
    hex_centers, ax1 = create_hex_grid(nx=100, ny=100, do_plot=True, align_to_origin = False, h_ax = ax1)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]
    loc2 = value_counts(runs,2)
    loc3 = value_counts(runs,3)
    loc4 = value_counts(runs,4)

    locs = ((loc2+loc3+loc4)/(3*len(runs)))

    if convolve:
        locs = Convolve(locs, 3, 0.75)
        np.save('con_locs_{}.npy'.format(A), np.asanyarray(locs))
    X = np.linspace(0,100,100)
    Y = np.linspace(0, max(y), 100)
    sin_z = [[sinusoid2D(X[i], Y[j], A, 0.1, 1) for i in range(100)] for j in range(100)]
    loc_plot = ax1.scatter(x,y,marker = 'h', s=17, c = locs, cmap = 'ocean') # gist_gray, gnuplot
    fig.colorbar(loc_plot, ax = ax1, shrink = 0.6)

    levels = np.linspace(min(sin_z), max(sin_z), 5)

    ax1.contour(X,Y, sin_z, levels = 5, cmap = 'binary', alpha = 0.6)#, levels)
        
    fig.suptitle('Heatmaps of location of AF induction \nand the Corresponding Coupling Space', fontsize = 16)
    plt.tight_layout()
    name = 'contour_heatmap_' + str(A) +'.png'
    if convolve:
        name = 'convolved_contour_heatmap_' + str(A) +'.png'
    plt.savefig(name)
    plt.close()

    

def Convolve(c,l,theta):
    #for a given index - calculate all indexs that should for convolve with it
    #With these indicies, calculate the exponential weight terms
    #With these uou can calculate the convolved value.
    #Scaling our vectors by theta - should scale length scale accordingly
    convolved = []
    coords = np.array([index_to_xy(i) for i in range(10000)])
    cnt = 0
    for i in coords:
        if cnt % 1000 == 0:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
        vals = np.array([euclidean(k, i) for k in coords])
        inc_bool = vals <= l
        c_diff = np.abs(c[inc_bool] - c[cnt])
        unsum = gaussian(vals[inc_bool], c_diff, theta)
        convolved.append(sum(unsum*c[inc_bool])/sum(unsum))
        cnt += 1
    return convolved


t0 = time.time()
plot_heat_map_contour('FailureMultiplierData_5', False) #0,1,3,5,10,20
t1 = time.time()
print(t1-t0)


#Amps and offsets
#(0.2,0.75)