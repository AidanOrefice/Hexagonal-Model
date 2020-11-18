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
import random
from configuration import config, title

def choose_numbers(list1, prob):
    new_list = []
    deleted_list = []
    for i in list1:
        p = random.uniform(0,1)
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
     graph = False, FullStateSave = False, settings_name = 'hi'):
        self.height = height
        self.width = width
        self.dt = 1 #Discrete time width of lattice.
        self.threshold = threshold
        self.runtime = runtime #Total time we want to run for.
        self.sig_st = sigmoid_strength
        self.coupling = pow(coupling, 1/2)
        self.ref_per = refractory_period + 2
        self.graph = graph
        self.settings = settings_name
        self.full_save = FullStateSave
        self.save_width = 300
        self.pacing_period = 75

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

    def Remove_random_bonds(self, n):
        for i in range(n):
            key = random.choice(list(self.neighbours))
            while len(self.neighbours[key]) == 0:
                key = random.choice(list(self.neighbours))
            else:
                neighbour = np.array([random.choice(list(self.neighbours[key]))])
                self.neighbours[key] = np.setdiff1d(self.neighbours[key],neighbour)
                self.neighbours[neighbour[0]] = np.setdiff1d(self.neighbours[neighbour[0]],key)
    
    def sinusoid2D(self, x, y, A1=1, A2=1, B1=0.25, B2=0.25, C1=0, C2=0, alpha = 0.01, beta = 0.7):
        #A - set max value of function
        #B - more/less peaks- stretches or compresses the peaks
        #C - phase shift everything 
        #applied BCs.
        return alpha * abs(A1 * np.sin(B1 * x + C1) + A2 * np.sin((B2 * y)*(2*np.pi)/ self.index_to_xy(self.height* self.width -1)  + C2)) + beta
    
    """  
    def CoupleDel(self):
        '''
        Have dictionary with neighbours. If we take a neighbour away from 1 say 2, need to take 1 from 2 as well
        '''
        keys = self.neighbours.keys()
        new_dic = {i : [] for i in range(len(keys))}
        deleted_dic = {}
        for i in keys:
            neighbours = self.neighbours[i]
            new, deleted =  choose_numbers(neighbours, self.coupling)
            self.neighbours[i] = new
            deleted_dic[i] = deleted
        
        for i in deleted_dic.keys():
            neighbours = deleted_dic[i]
            for j in neighbours:
                neighbours1 = list(self.neighbours[j])
                if i in neighbours1:
                    index = neighbours1.index(i)
                    neighbours2 = np.delete(neighbours1, index)
                    self.neighbours[j] = neighbours2


    def GradientMethodCoupling(self, start, end):
        delta = (end-start)/self.width

        keys = self.neighbours.keys()
        new_dic = {i : [] for i in range(len(keys))}
        deleted_dic = {}
        for i in keys:
            x,y = self.index_to_xy(i)
            grad_coupling = np.sqrt((delta*x) + start) 
            neighbours = self.neighbours[i]
            new, deleted =  choose_numbers(neighbours, grad_coupling)
            self.neighbours[i] = new
            deleted_dic[i] = deleted
        
        for i in deleted_dic.keys():
            neighbours = deleted_dic[i]
            for j in neighbours:
                neighbours1 = list(self.neighbours[j])
                if i in neighbours1:
                    index = neighbours1.index(i)
                    neighbours2 = np.delete(neighbours1, index)
                    self.neighbours[j] = neighbours2
    """
    


    def CouplingMethod(self, constant = False, gradient = False, norm_modes = True, start = 0.9 , end = 0.7):
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
                grad_coupling = np.sqrt(self.sinusoid2D(x,y))
                neighbours = self.neighbours[i]
                new, deleted =  choose_numbers(neighbours, grad_coupling)
                self.neighbours[i] = new
                deleted_dic[i] = deleted            
            pass

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

    def RunIt(self):
        self.t = 0
        self.RefHistory = np.zeros((self.save_width)  * len(self.ref), dtype = np.int16)
        self.AF = np.zeros(self.runtime, dtype = np.int16)
        i = 0
        j = 0
        done = True
        while self.t < self.runtime:
            if self.t == 0:
                self.Initialise()
                self.ActivationCheck()
                self.AF[0] = len(self.index_act)
                self.ChargeProp()
                if self.full_save:
                    self.RefHistory[0:len(self.ref)] = self.ref
                i += 1
                self.StateDevelop()
                self.t += self.dt
            elif self.t % self.pacing_period == 0:
                self.Initialise()
            self.ActivationCheck()
            self.AF[self.t] = len(self.index_act)
            count_last_100 = np.sum(self.AF[self.t-100:self.t])
            print(count_last_100)
            if (count_last_100 > 1.1 * self.height * len(self.AF[self.t-100:self.t]) and done):
                print('yassss')
                j = 1
                done = False
            if j == (self.save_width - 150):
                print('saving', self.t, i)
                np.save(title + 'i_{}'.format(i) + '.npy', self.RefHistory)
                j += 1
            elif j > 0:
                j += 1
            self.ChargeProp()
            if self.full_save:
                if i < self.save_width:
                    self.RefHistory[i*len(self.ref):(i+1)*len(self.ref)] = self.ref
                    i += 1
                else:
                    self.RefHistory[0:len(self.ref)] = self.ref
                    i = 1
            self.StateDevelop()
            self.t += self.dt
        if self.graph:
            f, ax = plt.subplots()
            x = [i for i in range(len(self.AF))]
            ax.plot(x, self.AF, ls = '-', label = 'Number of activated sites')
            ax.set_ylabel("Number of activated cells")
            ax.set_xlabel("Time")
            plt.savefig(self.settings + '.png')
        if self.full_save:
            np.save('StateData.npy', self.RefHistory)#Basically the same as below, only save interesting bits
            np.save('AF_timeline.npy', self.AF)#We won't save this, run statistics off this or maybe in code, good first spot

def main():
    t0 = time.time()

    np.random.seed(config['seed'])

    lattice = HexagonalLattice(config['width'],
        config['height'],
        config['runtime'],
        config['threshold'],
        config['sigmoid_strength'],
        config['coupling'],
        config['refractory_period'],
        config['graph'],
        config['FullStateSave'],
        title)
    
    lattice.CreateLattice()
    lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'],
     config['grad_start'], config['grad_end'] )
    lattice.RunIt()
    t1 = time.time()
    print('Runtime = %f s' % (t1-t0))

if __name__ == '__main__':
    main()




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
