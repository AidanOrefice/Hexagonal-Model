import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
from scipy.spatial.distance import euclidean
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
import time
import matplotlib as mpl

mpl.rcParams.update({
    'figure.figsize' : [12,9],
    'xtick.labelsize' : 15,
    'ytick.labelsize' : 15,
    'axes.labelsize' : 25,
    'legend.fontsize' : 17,
    'savefig.bbox' : 'tight',
})

def sinusoid2D(x, y, A,  amp, mean):
    #Amplitude - directly sets the amplitude of the function
    #Mean - directly sets the offset/mean of the function.
    # 0 < (Mean +/- amp) < 1 
    #A stretch out the modes. A must be an integer value.
    return (amp/2)*(np.sin(A*x*(2*np.pi/100))+np.sin(A*y*(2*np.pi/(100  - 100*(1-(np.sqrt(3)/2)))))) + mean #(x+25), (y+20) - ideal unit cell

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

def plot_heat_map(fname, fig, ax, convolve = False, presave = True, contour = False):
    runs = pd.read_csv(fname + '.csv')
    #Fix problem of trues in location 4, set to location 3 value
    runs.loc[runs['location_4'] == 'True', 'location_4'] = runs.loc[runs['location_4'] == 'True']['location_3']
    runs = runs.loc[runs['in AF?']] #Only look at ones that enter AF.

    A = int(fname.split('_')[-1])
    print(A)
    off = float(fname.split('_')[1])
    #101,201,301
    #runs = runs[runs['location_err']] Taken out to do earlier data

    if contour:
        ax1 = ax[0]
    else:
        ax1, ax2 = ax[0], ax[1]

    hex_centers, ax1 = create_hex_grid(nx=100, ny=100, do_plot=True, align_to_origin = False, h_ax = ax1)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]

    l_dict = {1:7, 3:5, 5:3 ,10: 1, 20: 1}

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
            print('connin')
            locs = Convolve(locs, l_dict[int(A)], 0.75)
            np.save('varkern_con_locs_{}.npy'.format(A), np.asanyarray(locs))
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
        ax1.tick_params(axis = 'both')

        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(loc_plot, cax = cax1, shrink = 1, ticks = [0,0.0003,0.0005]) #[round(min(locs),4),round(max(locs)/2,4), round(,4)*0.95]
        #cbar.ax.set_yticklabels([str(round(min(locs),4)),str(round(max(locs)/2,4)), str(round(max(locs),4))], fontsize = 16)  # vertically oriented colorbar


        sin_z = [sinusoid2D(x[i], y[i], A, 0.2, 0.5) for i in range(len(x))]
        _, ax2 = create_hex_grid(nx=100,ny=100, do_plot=True, align_to_origin = False, h_ax = ax2)
        ax2.axes.yaxis.set_ticklabels([])
        ax2.tick_params(axis = 'both', labelsize = 16)

        couple_plot = ax2.scatter(x,y,marker = 'h', s=17, c = sin_z)

        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(couple_plot, cax = cax2, shrink = 1, ticks = [min(sin_z),(max(sin_z)+min(sin_z))/2, max(sin_z)], pad = 0)
        #cbar.ax.set_yticklabels(['LOW', 'HIGH'], fontsize = 16)  # vertically oriented colorbar

        name = 'poster_heatmap_{}_'.format(off) + str(A) +'.png'
        #fig.suptitle('A = {}'.format(str(A)), fontsize =20, x  = 0.09, y = 0.91)
        
    #plt.tight_layout()
    if convolve:
        name = 'varkern_convolved_' + name
    #plt.savefig(name)
    #plt.close()
    
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

#Generates a mask of indicies and the unit cell to use.
def UnitCell(A, w, h):
    arr = []
    cellw = int(w/A)
    cellh = int(h/A)
    for i in range(cellh):
        initial = [j for j in range(i*cellw, (i*cellw) + cellw)] *A
        arr.extend(initial)
    arr = arr * A
    return arr, [0]*cellw*cellh

def UnitCellGenerate(fname):
    runs = pd.read_csv(fname + '.csv')
    #Fix problem of trues in location 4, set to location 3 value
    runs.loc[runs['location_4'] == 'True', 'location_4'] = runs.loc[runs['location_4'] == 'True']['location_3']
    runs = runs.loc[runs['in AF?']] #Only look at ones that enter AF.

    A = int(fname.split('_')[-1])
    print(A)
    off = float(fname.split('_')[1])
    #runs = runs[runs['location_err']] Taken out to do earlier data

    fig,ax = plt.subplots()

    hex_centers, ax = create_hex_grid(nx=int(100/A), ny=int(100/A), do_plot=True, align_to_origin = False, h_ax = ax)
    x = [i[0] for i in hex_centers]
    y = [i[1] for i in hex_centers]


    loc2 = value_counts(runs,2)
    loc3 = value_counts(runs,3)
    loc4 = value_counts(runs,4)
    locs = (loc2+loc3+loc4)/(3*len(runs))
    locs[101] = 0
    locs[201] = 0
    locs[301] = 0

    mask, unit = UnitCell(A,100,100)

    for ind,val in enumerate(locs):
        unit[mask[ind]] += val

    sdict = {1: 17, 5: 170, 10: 450, 20: 1300}
    unit_plot = ax.scatter(x,y,marker = 'h', s=sdict[A], c = unit, cmap = 'gnuplot2') # gist_gray, gnuplot
    ax.tick_params(axis = 'both', labelsize = 16)

    cbar = fig.colorbar(unit_plot, ax = ax, shrink = 0.9)
    '''divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(loc_plot, cax = cax1, shrink = 0.6, ticks = [min(unit), max(unit)])
    cbar.ax.set_yticklabels(['RARE', 'COMMON'], fontsize = 16)  # vertically oriented colorbar'''

    plt.savefig('UnitCell_{}'.format(A))
    plt.close()


def ReportFig():
    fig,(axi,axis) = plt.subplots(2,2)
    ax1, ax2 = axi[0], axi[1]
    ax3, ax4 = axis[0], axis[1]
    plot_heat_map('FailureMultiplierData_1', fig, [ax1,ax2], 1,1,0)
    plot_heat_map('FailureMultiplierData_3', fig, [ax3,ax4], 1,1,0)
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()

    fig.add_subplot(211, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title('A = 1:', fontsize = 16, loc = 'left')

    fig.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title('A = 3:', fontsize = 16, loc = 'left')

    plt.savefig('testie.png')

    #A=1 
    #A=3




if __name__ == '__main__':
    t0 = time.time()
    ReportFig()
    t1 = time.time()
    print(t1-t0)





''' Other Runs
plot_heat_map('FailureMultiplierData_1',1,1,0) #0,1,3,5,10,20

for i in ['FailureMultiplierData_0.4_full', 'FailureMultiplierData_0.5_full', 'FailureMultiplierData_0.6_full','FailureMultiplierData_0.7_full','FailureMultiplierData_0.8_full']:
    #data, convolve, presave, contour
    plot_heat_map(i,1,1,0) #0,1,3,5,10,20'''

