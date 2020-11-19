import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from configuration import title

def Activated_sites():
    global F
    f, ax = plt.subplots()
    #F is a 2D matrix, time in y-axis
    #A site is activated if its 1
    thing = []
    tot = 2500
    for i in range(int(F.shape[0]/tot)):
        data = F[i*tot:(i+1)*tot]
        unique, counts = np.unique(data, return_counts=True)
        x = dict(zip(unique, counts))
        if 1 in unique:
            thing.append(x[1])
        else:
            thing.append(0)
    x = [i for i in range(len(thing))]
    ax.plot(x, thing, ls = '-', label = 'Number of activated sites')
    ax.set_ylabel("Number of activated cells")
    ax.set_xlabel("Time")
    #f = open('settings.txt', 'r')
    #settings = f.read()
    plt.savefig(title + str('.png'))


if __name__ == "__main__":
    global F
    F = np.load('StateData.npy')
    Activated_sites()