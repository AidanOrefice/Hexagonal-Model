import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Hexagon import HexagonalLattice
from hexalattice.hexalattice import *
from CouplingViz import sinusoid2D
from scipy.spatial.distance import euclidean
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_heat_map(fname, convolve = False, presave = True, contour = False,):
    runs = pd.read_csv(fname + '.csv')
    #Fix problem of trues in location 4, set to location 3 value
    runs.loc[runs['location_4'] == 'True', 'location_4'] = runs.loc[runs['location_4'] == 'True']['location_3']
    runs = runs.loc[runs['in AF?']] #Only look at ones that enter AF.

    A = 1#int(fname.split('_')[-1])
    off = float(fname.split('_')[1])
    #101,201,301
    runs = runs[runs['location_err']]

    if contour:
        fig,ax1 = plt.subplots()
    else:
        fig,(ax1,ax2) = plt.subplots(1,2)
        fig.set_size_inches(16,8)

    hex_centers, ax1 = create_hex_grid(nx=100, ny=100, do_plot=True, align_to_origin = False, h_ax = ax1)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]

    if presave:
        print('yes')
        locs = np.load('con_locs_{}.npy'.format(A)) #Retri
    else:
        print('errr')
        loc2 = value_counts(runs,2)
        loc3 = value_counts(runs,3)
        loc4 = value_counts(runs,4)

        locs = (loc2+loc3+loc4)/(3*len(runs))

        if convolve:
            locs = Convolve(locs, 3, 0.75)
            np.save('con_locs_{}.npy'.format(A), np.asanyarray(locs))
    locs[101] = 0
    locs[201] = 0
    locs[301] = 0

    if contour:
        loc_plot = ax1.scatter(x,y,marker = 'h', s=17, c = locs, cmap = 'pink') # gist_gray, gnuplot, gist_rainbow, ocean, jet
        plt.title('Heatmaps of location of AF induction \nand the Corresponding Coupling Space, A = {}'.format(str(A)), fontsize =13)
        
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(loc_plot, cax = cax1, shrink = 0.6, ticks = [min(locs), max(locs)])
        cbar.ax.set_yticklabels(['RARE', 'COMMON'])  # vertically oriented colorbar

        X = np.linspace(0,100,100)
        Y = np.linspace(0, max(y), 100)
        sin_z = [[sinusoid2D(X[i], Y[j], A, 0.1, 1) for i in range(100)] for j in range(100)]

        levels_dict = {'1': 5, '3': 5, '5': 5, '10': 3, '20': 3 }
        alpha_dict = {'1': 1, '3': 0.8, '5': 0.6, '10': 0.5, '20': 0.5}

        cp = ax1.contour(X,Y, sin_z, levels = levels_dict[str(A)], cmap = 'gray', alpha = alpha_dict[str(A)])# change the number of levels, binary,gray
        name = 'contour_heatmap_' + str(A) +'.png'

        mini = np.where(sin_z == np.min(sin_z))
        min_x, min_y = x[mini[0][0]], Y[mini[1][0]]
        maxi = np.where(sin_z == np.max(sin_z))
        max_x, max_y = X[maxi[0][0]], Y[maxi[1][0]]

        print(min_x,min_y)
        print(max_x,max_y)

        #Adding labels to the max and min
        if str(A) == '1': 
            ax1.text(max_x-4, max_y, 'HIGH', color = 'white')
            ax1.text(min_x-4, min_y, 'LOW', color = 'white')

        '''print(cp.levels)
        fmt = {}
        strs = ['low','high']
        for l, s in zip(cp.levels, strs):
            fmt[l] = s

        # Label every other level using strings
        ax1.clabel(cp, [cp.levels[0], cp.levels[-1]], inline=True, fmt=fmt, fontsize=10)'''

    else:
        loc_plot = ax1.scatter(x,y,marker = 'h', s=17, c = locs, cmap = 'gnuplot2') # gist_gray, gnuplot

        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(loc_plot, cax = cax1, shrink = 0.6, ticks = [min(locs), max(locs)])
        cbar.ax.set_yticklabels(['RARE', 'COMMON'])  # vertically oriented colorbar


        sin_z = [sinusoid2D(x[i], y[i], A, 0.1, 1) for i in range(len(x))]
        _, ax2 = create_hex_grid(nx=100,ny=100, do_plot=True, align_to_origin = False, h_ax = ax2)

        couple_plot = ax2.scatter(x,y,marker = 'h', s=17, c = sin_z)

        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(couple_plot, cax = cax2, shrink = 0.6, ticks = [min(sin_z), max(sin_z)])
        cbar.ax.set_yticklabels(['LOW', 'HIGH'])  # vertically oriented colorbar

        name = 'heatmap_{}_'.format(off) + str(A) +'.png'
        fig.suptitle('Heatmaps of location of AF induction \nand the Corresponding Coupling Space, A = {}'.format(str(A)), fontsize =20)
        
    plt.tight_layout()
    if convolve:
        name = 'convolved_' + name
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

def ReturnUnitCell(a, p):
    width = 100
    height = 100
    hex_centers, _ = create_hex_grid(nx=100, ny=100, do_plot=False, align_to_origin = False)
    bar_ind = [round((width*i)/p)-1 for i in range(1,p+1)]
    bar_xy = [index_to_xy(i) for i in bar_ind]
    print(bar_xy)
    
    ind = 0
    for i in range(1,p+1):
    #a - array of data for the locations (should not be convolved.)
    #p - periodicity

    #Using the periodicity need to setup unit cell widths and heights (i.e. boundary locations) which I will then use 
    #to split up a.

    #Try and do a unit cell around the maximum site locations
    #Create a mask indicating what cell each site belongs to 

        pass


t0 = time.time()
'''
for i in ['FailureMultiplierData_0.4_full', 'FailureMultiplierData_0.5_full', 'FailureMultiplierData_0.6_full','FailureMultiplierData_0.7_full','FailureMultiplierData_0.8_full']:
    plot_heat_map(i,0,0,0) #0,1,3,5,10,20'''
a = []
ReturnUnitCell(a,3)
t1 = time.time()
print(t1-t0)


#Amps and offsets
#(0.2,0.75)


'''
#Applying limits to the runs - DONT NEED THIS ANYMORE
for ind, row in runs.iterrows():
    list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
    amp, offs = float(list_[1]), float(list_[2])
    if (amp > 0.5) or (amp < 0.05) or (offs < 0.59):
        runs = runs.drop(index = ind)'''
