"""
Start Date: 12/10/2020
Authors: Daniel Loughran and Aidan Orefice
Script to initialise the hexagonal lattice

Create a simple animation to highlight the basic mechanics of lattice are working. 
"""

import numpy as np
import time
import random
from line_profiler import LineProfiler
import matplotlib.pyplot as plt
from configuration import *
from Animation import Where_reentry
from hexalattice.hexalattice import *
import pandas as pd
from itertools import groupby
from operator import itemgetter

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

'''def Remove_random_bonds(self, n):
    for i in range(n):
        key = random.choice(list(self.neighbours))
        while len(self.neighbours[key]) == 0:
            key = random.choice(list(self.neighbours))
        else:
            neighbour = np.array([random.choice(list(self.neighbours[key]))])
            self.neighbours[key] = np.setdiff1d(self.neighbours[key],neighbour)
            self.neighbours[neighbour[0]] = np.setdiff1d(self.neighbours[neighbour[0]],key)'''

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
        self.pacing_period = 75
        self.stats = stats
        
        if seed == 0:
            self.seed = np.random.randint(0,int(1e7))
        else:
            self.seed = seed
        np.random.seed(self.seed)

        if self.full_save == 'full':
            self.save_width = self.runtime
        elif self.full_save == 'transition':
            self.save_width = 300
        else:
            self.save_width = self.full_save

        '''#Initialise dataframe to save each run
        columns = list(config.keys())
        columns.append('seed')
        self.df = pd.DataFrame(columns = columns)'''

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
    
    def sinusoid2D(self, x, y,  amp, mean, A1 = 0.25, A2 = 0.25):
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
            self.coupling_sample = [len(self.neighbours[i])/6 for i in self.neighbours.keys()]

        for i in deleted_dic.keys():
            neighbours = deleted_dic[i]
            for j in neighbours:
                neighbours1 = list(self.neighbours[j])
                if i in neighbours1:
                    index = neighbours1.index(i)
                    neighbours2 = np.delete(neighbours1, index)
                    self.neighbours[j] = neighbours2


    def Initialise(self):
        self.index_int = [i*self.width for i in range(self.height)] #Left hand side
        #self.hexagon[index_init] = 100 
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

    def in_AF(self):
        #print(len(self.index_act))
        if len(self.index_act) > self.height * 1.1:
            return True
        else:
            return False

    '''def save_df(self):
        return run
        self.df.loc[len(self.df)] = run'''
    
    def trans_save(self,i,j):
        count_last_100 = np.sum(self.AF[self.t-100:self.t])
        if (count_last_100 > 1.1 * self.height * len(self.AF[self.t-100:self.t])):
            j = 1
        if j == (self.save_width - 150):
            print('saving', self.t, i)
            #np.save(title + 'i_{}'.format(i) + '.npy', self.RefHistory)
            #self.save_df(Where_reentry_whole(self.RefHistory))
            j = 999
        elif j > 0:
            print('hi')
            j += 1
        if i < self.save_width:
            self.RefHistory[i*len(self.ref):(i+1)*len(self.ref)] = self.ref
            i += 1
        else:
            self.RefHistory[0:len(self.ref)] = self.ref
            i = 1
        return i,j

    def length_save(self, i):
        if i < self.save_width:
            self.RefHistory[i*len(self.ref):(i+1)*len(self.ref)] = self.ref
            i += 1
        else:
            self.RefHistory[0:len(self.ref)] = self.ref
            i = 1
        return i

    def RunIt(self):
        self.t = 0
        self.sites_found = {}
        if self.full_save != False:
            self.RefHistory = np.zeros(((self.save_width)  * len(self.ref)), dtype = np.int16)
        self.AF = np.zeros(self.runtime, dtype = np.int16)
        i = 0
        joke = 0
        found = 0
        print(self.seed)
        while self.t < self.runtime:
            if self.t == 0:
                self.Initialise()
                self.ActivationCheck()
                self.AF[0] = len(self.index_act)
                self.ChargeProp()
                if self.full_save != False:
                    self.RefHistory[0:len(self.ref)] = self.ref
                i += 1
                self.StateDevelop()
                self.t += self.dt
            elif self.t % self.pacing_period == 0:
                self.Initialise()
            self.ActivationCheck()
            if self.full_save == 'full':
                self.RefHistory[self.t*len(self.ref):(self.t+1)*len(self.ref)] = self.ref
            elif self.full_save == 'transition':
                i,joke = self.trans_save(i,joke)
                if joke == 999:
                    print('done')
                    return self.RefHistory, True, i
            elif self.full_save == False:
                pass
            else:
                i = self.length_save(i)
            self.AF[self.t] = len(self.index_act)
            self.ChargeProp()
            self.StateDevelop()
            self.t += self.dt
        if self.graph:
            f, ax = plt.subplots()
            x = [i for i in range(len(self.AF))]
            ax.plot(x, self.AF, ls = '-', label = 'Number of activated sites')
            ax.set_ylabel("Number of activated cells")
            ax.set_xlabel("Time")
            plt.savefig('SetThisAsTheSettings' + '.png')  #################################
        if self.full_save == 'full':
            np.save('StateData.npy', self.RefHistory)#Basically the same as below, only save interesting bits
            #np.save('AF_timeline.npy', self.AF)#We won't save this, run statistics off this or maybe in code, good first spot

        # return the settings of each run
        run = list(config.values())
        run.append(self.seed)
        return run

def InitialDF():
    columns = list(config.keys())
    columns.extend(['seed', 'in AF?', '%time in AF'])
    df = pd.DataFrame(columns=columns)
    return df

def NormalModes():
    t0 = time.time()
    df = InitialDF()
    amps = np.linspace(0,0.5,21)
    means = np.linspace(0,1,21)
    print(means)
    print(amps)
    for k in means:
        for i in amps:
            print(i)
            for j in range(51):
                
                lattice = HexagonalLattice(config['width'],
                    config['height'],
                    config['runtime'],
                    config['threshold'],
                    config['sigmoid_strength'],
                    config['coupling'],
                    config['refractory_period'],
                    config['graph'],
                    config['FullStateSave'],
                    config['stats'],
                    config['set_seed'])
                
                lattice.CreateLattice()
                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [i,k],
                config['grad_start'], config['grad_end'] )
                run = lattice.RunIt()
                run[13] = [0.25,1,i,k]

                index = np.where(lattice.AF > 55)[0]
                thing = [list(map(itemgetter(1), g)) for tk, g in groupby(enumerate(index), lambda ix : ix[0] - ix[1])]
                thing = [i for i in thing if len(i) > 10]
                len_thing = 0
                if len(thing) > 0:
                    in_AF = True
                    for x in thing:
                        len_thing += len(x)
                else:
                    in_AF = False

                fraction_in_AF = len_thing/config['runtime']

                run.extend([in_AF, fraction_in_AF]) 

                df.loc[len(df)] = run
    df.to_csv('Trial_Varying_Variance_PS_0.25.csv')

    t1 = time.time()
    print('Runtime = %f s' % (t1-t0))
    

def main():
    t0 = time.time()
    print(config['normal_modes_config'])
    lattice = HexagonalLattice(config['width'],
        config['height'],
        config['runtime'],
        config['threshold'],
        config['sigmoid_strength'],
        config['coupling'],
        config['refractory_period'],
        config['graph'],
        config['FullStateSave'],
        config['stats'],
        config['set_seed'])
    
    lattice.CreateLattice()
    lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], config['normal_modes_config'],
     config['grad_start'], config['grad_end'] )
    lattice.RunIt()

    fig,ax = plt.subplots()
    x = [lattice.index_to_xy(i)[0] for i in range(2500)]
    y = [lattice.index_to_xy(i)[1] for i in range(2500)]
    a = ax.scatter(x,y,marker = 'h', s=17, c = lattice.coupling_sample)
    fig.colorbar(a,shrink=0.75)
    plt.title('Sample of -0.25*[sin(0.25x) + sin(2pi*y/50)]+0.85')
    plt.savefig('CouplingShow.png')

    t1 = time.time()
    print('Runtime = %f s' % (t1-t0))

if __name__ == '__main__':
    NormalModes()




'''seed = int(1e6)
np.random.seed(seed)
width = 50
height = 50
runtime = 1000
threshold = 0.2
sigmoid_strength = 20
coupling = 0.7
lattice = HexagonalLattice(width,height,runtime,threshold,sigmoid_strength, coupling)
print("Width is:", str(width) + ", Height is:", str(height))
f = open('settings.txt', 'w')
f.write(str(width) + "," + str(height) + "," + str(runtime) + "," + str(threshold) + "," + str(sigmoid_strength) + "," + str(coupling) + "," + str(seed))
lattice.CreateLattice()
lattice.CoupleDel()

lp = LineProfiler()
lp_wrapper = lp(lattice.RunIt)
lp_wrapper()
lp.print_stats()'''
