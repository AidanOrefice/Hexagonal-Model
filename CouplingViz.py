import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from hexalattice.hexalattice import *
from configuration import config


"""
Script that visualises the coupling in the lattice.

Uses https://github.com/alexkaz2/hexalattice/blob/master/hexalattice/hexalattice.py
"""
amp = 0.5
mean = 0.2
A1 = 0.25
A2 = 1

def sinusoid2D(x, y, A1=A1, A2=A2,  amp = amp, mean = mean):
    #Amplitude - directly sets the amplitude of the function
    #Mean - directly sets the offset/mean of the function.
    # 0 < (Mean +/- amp) < 1 
    #A1/A2 stretch out the modes.
    #A2 must be an integer value to ensure periodicity.
    return (amp/2)*(np.sin(A1*x)+np.sin(A2*y*(2*np.pi/index_to_xy(2499)[1]))) + mean

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

fig,ax = plt.subplots()

hex_centers, ax = create_hex_grid(nx=50,ny=50, do_plot=True, align_to_origin = False, h_ax = ax)
x = [i[0] for i in hex_centers]
y = [i[1] for i in hex_centers] 

sin_z = [sinusoid2D(x[i], y[i]) for i in range(len(x))]
grad_z = [gradient(i) for i in x]
a = ax.scatter(x,y,marker = 'h', s=17, c = sin_z)
fig.colorbar(a,shrink=0.75)

print(np.mean(sin_z))
print(np.var(sin_z))
print(np.std(sin_z))

label_mean = 'Mean = ' + str(mean)
label_amp = 'Amplitdue = ' + str(amp)

legend_elements = [Line2D([0], [0], marker='o', color='white', label=label_mean, markerfacecolor='white', markersize=0),
            Line2D([0], [0], marker='o', color='white', label=label_amp, markerfacecolor='white', markersize=0)]



plt.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

plt.title(r"$\frac{Amplitude}{2} \times \left( \sin(\frac{x}{4}) + \sin(\frac{2\pi y}{height}) \right) + Mean$", fontsize = 16)

plt.savefig('viz_test.png')

""""
leng = np.arange(2500)
x= [index_to_xy(i)[0] for i in leng]
y= [index_to_xy(i)[1] for i in leng]



x, y = np.meshgrid(np.arange(1,51), np.arange(1,51))
z = [sinusoid2D(x[i], y[i]) for i in range(len(x))]
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(x, y, z, 60, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

"""