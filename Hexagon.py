"""
Start Date: 12/10/2020
Authors: Daniel Loughran and Aidan Orefice
Script to initialise the hexagonal lattice

Create a simple animation to highlight the basic mechanics of lattice are working. 
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from configuration import *
import pandas as pd
from itertools import groupby
from operator import itemgetter
from matplotlib.lines import Line2D

def choose_numbers(list1, prob):
    new_list = []
    deleted_list = []
    for i in list1:
        p = np.random.uniform(0,1)
        if p < prob:
            new_list.append(i)
        else:
            deleted_list.append(i)
    return new_list, deleted_list

class HexagonalLattice():
    '''
    Class that controls the physical attributes of the hexagonal lattice.

    Parameters:
    height - vertical length of the hexagonal lattice, in units of # of cells.
    width - horizontal length of the vertical lattice, in units of # of cells.

    Attributes:
    height - vertical length of the hexagonal lattice, in units of # of cells.
    width - horizontal length of the hexagonal lattice, in units of # of cells.
    hexagon - hexagonal lattice unrolled into 1d array - used for charge spread.
    ref - hexagonal lattice unrolled into 1d array - used for keeping track of state of cell.
    neighbours - dictionary of the neighbours of each lattice site.
    '''
    def __init__(self, width, height, runtime, threshold, sigmoid_strength, coupling = 1, refractory_period = 10,
     graph = False, FullStateMethod = 'full', stats = False, seed = 0):
        self.height = height
        self.width = width
        self.dt = 1 #Discrete time width of lattice.
        self.threshold = threshold
        self.runtime = runtime #Total time we want to run for.
        self.sig_st = sigmoid_strength
        self.coupling = pow(coupling, 1/2)
        self.ref_per = refractory_period + 2
        self.graph = graph
        self.full_save = FullStateMethod #Options r full (whole run), any number (last x timesteps), transition (150 before AF, 150 after AF), False (Nothign saved)
        self.pacing_period = width * 2
        self.stats = stats
        
        if seed == 0:
            self.seed = np.random.randint(0,int(2**32 - 1))
        else:
            self.seed = seed
        np.random.seed(self.seed)

        if self.full_save == 'full':
            self.save_width = self.runtime
        elif self.full_save == 'transition':
            self.save_width = 150

        self.title = title + str(self.seed)

        #Ensuring lattice is of the correct dimensions - for toroidal geometry lattice must be even int x even int
        if not(self.width % 2 == 0) or not(self.height % 2 == 0):
            raise ValueError('The lattice must be of dimensions (even integer) x (even integer).')

    def CreateLattice(self):
        self.hexagon = np.zeros((self.width * self.height), dtype = np.float16) #Hexagonal Lattice in 1d array.
        self.ref = np.zeros((self.width * self.height), dtype = np.float16) #Identical for recording refractoriness.
        self.Neighbours()

    def CreateLatticeNoNeighbours(self):
        self.hexagon = np.zeros((self.width * self.height), dtype = np.float16) #Hexagonal Lattice in 1d array.
        self.ref = np.zeros((self.width * self.height), dtype = np.float16) #Identical for recording refractoriness.

    def Neighbours(self):
        """
        Method to initially identify the neighbours of each site on the hexagonal lattice.
        Only used during initialisation period of main script.


        Boundary Conditions:
        Currently modelled with a toroidal topology i.e. periodic in x and y plane.
        Next step could be to introduce a clindrical topology - this would mean create a discontinuity at the ends of
        the lattice in x plane. To implement this all j=0 or j =self.width - 1 sites, i.e all sites in the first and
        last columns, should be altered to not connect to sites on opposing side of lattice.


        Returns:
        neighbours - A dictionary with each lattice site (key) having a 1d array of it's lattice neighbours (value).
        """
        self.neighbours = {}
        for i in range(self.height):
            if i == 0: #First row always even
                for j in range(0, self.width):
                    if j == 0: # (0,0) root cell of the hexagonal lattice (bottom left).
                        self.neighbours[0] = np.asarray([self.width, 1,
                         self.width * (self.height -1)])
                         #, (self.width * 2) - 1, self.width - 1, self.width * self.height - 1
                    elif j == self.width - 1: #bottom right corner
                        self.neighbours[j] = np.asarray([j - 1, j + self.width - 1, j + self.width,
                         self.width * self.height - 1, self.width * self.height - 2])
                         #, 0
                    else:
                        self.neighbours[j] = np.asarray([j - 1, j + 1, j + self.width - 1, j + self.width,
                         self.width * (self.height - 1) + j - 1, self.width * (self.height - 1) + j])
            elif i == self.height - 1: #Last row always odd
                for j in range(0, self.width):
                    index = i * self.width + j
                    if j == 0: #top left corner
                        self.neighbours[index] = np.asarray([index - self.width, index - self.width + 1,
                         index + 1, 0, 1])
                         #, self.width * self.height - 1
                    elif j == self.width - 1: #top right corner
                        self.neighbours[index] = np.asarray([ index - self.width,
                         index - 1, self.width - 1])
                         #, index - self.width + 1, 0,index - (self.width * 2) + 1,
                    else:
                        self.neighbours[index] = np.asarray([index - self.width, index - self.width + 1,
                         index - 1, index + 1, j, j + 1])
            else: #All intermediate rows.
                if_even = i % 2 == 0 #False is odd True is even
                for j in range(0, self.width):
                    index = i * self.width + j
                    if j == 0: #left-most column
                        if if_even:
                            self.neighbours[index] = np.asarray([
                             index + self.width, index - self.width, index + 1])
                             #index - 1, index + self.width - 1,, index + (self.width * 2) - 1
                        else:
                            self.neighbours[index] = np.asarray([index + 1,
                             index + self.width + 1, index + self.width, index - self.width + 1, index - self.width])
                             # index + self.width - 1,
                    elif j == self.width - 1: #right-most column
                        if if_even:
                            self.neighbours[index] = np.asarray([ index - 1,
                             index + self.width - 1, index + self.width, index - self.width - 1, index - self.width])
                             #index - self.width + 1,
                        else:
                            self.neighbours[index] = np.asarray([index - 1,
                             index + self.width, index - self.width])
                             #, index + 1, index - (self.width * 2) + 1, index - self.width + 1
                    else: #All non-edge sites.
                        if if_even:
                            self.neighbours[index] = np.asarray([index + 1, index - 1, index + self.width - 1,
                             index + self.width, index - self.width - 1, index - self.width])
                        else:
                            self.neighbours[index] = np.asarray([index + 1, index - 1, index + self.width + 1,
                             index + self.width, index - self.width + 1, index - self.width])
    
    def index_to_xy(self, index):
        row = np.floor(index / self.width)
        y = row - row*(1-(np.sqrt(3)/2)) #fix.
        if_even = row % 2 == 0
        if if_even:
            x = index - (row * self.width)
        else:
            x = index - (row * self.width) + 0.5
        return (x,y)
    
    def sinusoid2D(self, x, y,  amp, mean, A1 = 0.25, A2 =1):
        #Amplitude - directly sets the amplitude of the function
        #Mean - directly sets the offset/mean of the function.
        # 0 < (Mean +/- amp) < 1 
        #A1/A2 stretch out the modes.
        #A2 must be an integer value to ensure periodicity.
        amp = float(amp)
        mean = float(mean)
        A1 = float(A1)
        A2 = float(A2)
        return (amp/2)*(np.sin(A1*x)+np.sin(A2*y*(2*np.pi/self.index_to_xy(self.height* self.width -1)[1]))) + mean

    def CouplingMethod(self, constant = False, gradient = False, norm_modes = True, sinusoid_params = [0.1,0.6], 
    start = 0.9 , end = 0.7):
        if constant + gradient + norm_modes != 1:
            raise ValueError('Cannot decouple using two different methods.')

        keys = self.neighbours.keys()
        copy = self.neighbours.copy()
        #new_dic = {i : [] for i in range(len(keys))}
        deleted_dic = {}

        if constant:
            for i in keys:
                neighbours = self.neighbours[i]
                new, deleted =  choose_numbers(neighbours, self.coupling)
                self.neighbours[i] = new
                deleted_dic[i] = deleted           
        
        elif gradient:
            delta = (end-start)/self.width
            for i in keys:
                x,y = self.index_to_xy(i)
                grad_coupling = np.sqrt((delta*x) + start) 
                neighbours = self.neighbours[i]
                new, deleted =  choose_numbers(neighbours, grad_coupling)
                self.neighbours[i] = new
                deleted_dic[i] = deleted

        elif norm_modes:
            for i in keys:
                x,y = self.index_to_xy(i)
                grad_coupling = np.sqrt(self.sinusoid2D(x, y, *sinusoid_params))
                neighbours = self.neighbours[i]
                new, deleted =  choose_numbers(neighbours, grad_coupling)
                self.neighbours[i] = new
                deleted_dic[i] = deleted            
            self.coupling_samp = [len(self.neighbours[i])/len(copy[i]) for i in self.neighbours.keys()]

        for i in deleted_dic.keys():
            neighbours = deleted_dic[i]
            for j in neighbours:
                neighbours1 = list(self.neighbours[j])
                if i in neighbours1:
                    index = neighbours1.index(i)
                    neighbours2 = np.delete(neighbours1, index)
                    self.neighbours[j] = neighbours2

    def Coupling_Sample(self, mean, amp):
        fig,ax = plt.subplots()
        x = [self.index_to_xy(i)[0] for i in range(2500)]
        y = [self.index_to_xy(i)[1] for i in range(2500)]
        a = ax.scatter(x,y,marker = 'h', s=17, c = self.coupling_samp, cmap=plt.cm.get_cmap('viridis', 7))
        cbar = plt.colorbar(a, ticks=np.arange(1/14,17/14,1/7), shrink = 0.75)
        cbar.ax.set_yticklabels(['0', '1/6', '1/3', '1/2', '2/3', '5/6', '1'])

        label_mean = 'Mean = ' + str(mean)
        label_amp = 'Amplitude = ' + str(amp)
        legend_elements = [Line2D([0], [0], marker='o', color='white', label=label_mean, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_amp, markerfacecolor='white', markersize=0)]
        plt.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.title(r"Sample of $\frac{Amplitude}{2} \times \left( \sin(\frac{x}{4}) + \sin(\frac{2\pi y}{height}) \right) + Mean$", fontsize = 16)
        plt.savefig('CouplingSpaceSample.png')
    
    def Initialise(self):
        self.index_int = [i*self.width for i in range(self.height)] #Left hand side
        self.ref[self.index_int] = 1

    def SigmoidDist(self,charges):
        return 1/(1+np.exp(-self.sig_st*(charges-self.threshold)))

    def ActivationCheck(self):
        index_charged = np.where(self.hexagon > 0)[0]
        p = self.SigmoidDist(self.hexagon[index_charged])
        a = np.random.rand(len(index_charged))
        self.index_act = index_charged[p>a]
        self.ref[self.index_act] = 1 #Set sites to activated.
        self.hexagon[index_charged] = 0
        if self.t % self.pacing_period == 0:
            self.index_act = np.concatenate((self.index_act,self.index_int))
        
    #Uses sites that have been set to activated and spreads their charge. Resets charge to zero of activated sites.
    def ChargeProp(self):
        #index_act = np.where(self.ref == 1)[0] #sites that are activated - need to spread their charge
        for ind in self.index_act:
            neighbours = self.neighbours[ind]
            avail_neighbours = [i for i in neighbours if self.ref[i] == 0]
            if len(avail_neighbours) > 0:
                self.hexagon[avail_neighbours] += 1/len(avail_neighbours)

    #Develops the states of each site.
    def StateDevelop(self):
        self.ref[self.ref >= 1] += 1
        self.ref[self.ref == self.ref_per] = 0

    def Per_check(self, per):
        indi = [(i*self.width) - 1 for i in range(1,self.height + 1)]#Indexs of right hand side
        pre_length = len(indi)
        for i in range(self.width - int(self.width/5), self.pacing_period):
            data = self.RefHistory[(self.t + i) * len(self.ref) : (self.t + i + 1) * len(self.ref)]
            indi = [i for i in indi if data[i] != 0]
            if len(indi) < pre_length / 4: #Arbitrary fraction
                self.per = True
                return [per[0] + 1, per[1] + 1]
        return [per[0], per[1] + 1]

    def Stats_check(self):
        if self.in_AF == False:
            self.percolating = self.Per_check(self.percolating)
            if self.AF_check():
                self.in_AF = True
                self.AF_bool.append((self.t, self.in_AF))
            else:
                self.AF_bool.append((self.t, self.in_AF))
        else:
            if self.AF_check():
                self.in_AF = True
            else:
                self.in_AF = False
            self.AF_bool.append((self.t, self.in_AF))

    def AF_check(self):
        if sum(self.AF[self.t - self.pacing_period:self.t]) > len(self.AF[self.t - self.pacing_period:self.t]) * self.height * 1.1:
            return True #If sum of activ. sites over pacing period > 1.1 * expected number of activate --> FIBRILLATION.
        width_dif = int((self.pacing_period - self.width) / 2)  #Checking if sites are activated at end of beat - trying to catch special cases
        avg_over_dif = sum(self.AF[self.t - width_dif:self.t]) / width_dif
        avg_over_norm = sum(self.AF[self.t + 1 - self.pacing_period:self.t - width_dif * 2]) / len(self.AF[self.t + 1 - self.pacing_period:self.t - width_dif * 2])
        if avg_over_dif > avg_over_norm / 6: #6 is an arbitrary fraction
            return True
        else:
            return False

    def Search_Meth2(self): #Searches for the 2nd, 3rd and 4th re-excited sites. Returns time of 2nd re-excitation.
        sites = {}
        re_sites = {2:True, 3:True, 4:True}
        for j in range(self.AF_first_beat - self.pacing_period + 1, self.AF_last_beat):
            time_data = self.RefHistory[j*len(self.ref):(j+1)*len(self.ref)]
            activated_sites = np.where(time_data == 1)[0]
            for i in activated_sites:
                if i in list(sites.keys()):
                    sites[i] += 1
                    if re_sites[2] == True:
                        if sites[i] == 2:
                            re_sites[2] = i
                            self.AF_time = (max(j-30,1), j+30)
                            print(self.AF_time)  #Transition time range
                    if re_sites[3] == True:
                        if sites[i] == 3:
                            re_sites[3] = i
                    if re_sites[4] == True:
                        if sites[i] == 4:
                            re_sites[4] = i
                            return re_sites
                else:
                    sites[i] = 1
        return re_sites

    def Graph(self):
        f, ax = plt.subplots()
        x = [i for i in range(len(self.AF))]
        ax.plot(x, self.AF, ls = '-', label = 'Number of activated sites')
        ax.set_ylabel("Number of activated cells")
        ax.set_xlabel("Time")
        plt.savefig('Graphed' + '.png')  #################################

    def save_choice(self): #Run once at end
        #AF Start time and location
        beat_af = [i[0] // self.pacing_period for i in self.AF_bool if i[1] == True]
        consec_AF_beats = [list(map(itemgetter(1), g)) for tk, g in groupby(enumerate(beat_af), lambda ix : ix[0] - ix[1])]
        consec_AF_beats_3 = [i for i in consec_AF_beats if len(i) > 2]
        self.AF_first_beat = consec_AF_beats_3[0][0] * 100  #first beat after fib starts
        self.AF_last_beat = consec_AF_beats_3[0][-1] * 100  #last beat after fib starts
        print(self.AF_first_beat, self.AF_last_beat)
        self.re_sites = self.Search_Meth2()  #2nd,3rd,4th activated sites.
        print(self.re_sites)

    def RunIt(self):
        self.t = 0
        self.RefHistory = np.zeros(((self.runtime)  * len(self.ref)), dtype = np.int16)
        self.AF = np.zeros(self.runtime, dtype = np.int16)
        self.done = True
        self.in_AF = False
        self.AF_bool = []
        print(self.seed)
        self.percolating = [0,0]
        while self.t < self.runtime:
            if self.t == 0:
                self.Initialise()
                self.ActivationCheck()
                self.AF[0] = len(self.index_act)
                self.ChargeProp()
                if self.full_save != False:
                    self.RefHistory[0:len(self.ref)] = self.ref
                self.StateDevelop()
                self.t += self.dt
            elif self.t % self.pacing_period == 0:
                self.Stats_check()
                self.Initialise()
            self.ActivationCheck()
            self.RefHistory[self.t*len(self.ref):(self.t+1)*len(self.ref)] = self.ref
            self.AF[self.t] = len(self.index_act)
            self.ChargeProp()
            self.StateDevelop()
            self.t += self.dt
        if self.graph:
            self.Graph()
        if self.in_AF:
            self.save_choice()
        if self.full_save == 'full':
            np.save(self.title + '.npy', self.RefHistory)
        elif self.full_save == 'transition' and self.in_AF:
            np.save(self.title + '.npy', self.RefHistory[self.AF_time[0] * len(self.ref):self.AF_time[1] * len(self.ref)])
        run = list(config.values())
        run.append(self.seed)
        #'location_2', 'location_3', 'location_4', 'AF_time',
        if self.in_AF:
            run.append(self.re_sites[2])
            run.append(self.re_sites[3])
            run.append(self.re_sites[4])
            run.append(self.AF_time)
        else:
            run.append(False)
            run.append(False)
            run.append(False)
            run.append(False)
        run.append(self.percolating[0] / self.percolating[1])
        run.append(self.title)
        return run





