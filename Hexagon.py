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
    def __init__(self,width,height, runtime, threshold, sigmoid_strength, coupling = 1, refractory_period = 10):
        self.height = height
        self.width = width
        self.dt = 1 #Discrete time width of lattice.
        self.threshold = threshold
        self.runtime = runtime #Total time we want to run for.
        self.sig_st = sigmoid_strength
        self.coupling = pow(coupling, 1/2)
        self.ref_per = refractory_period

        #Ensuring lattice is of the correct dimensions - for toroidal geometry lattice must be even int x even int
        if not(self.width % 2 == 0) or not(self.height % 2 == 0):
            raise ValueError('The lattice must be of dimensions (even integer) x (even integer).')

    def CreateLattice(self):
        self.hexagon = np.zeros((self.width * self.height), dtype = np.float16) #Hexagonal Lattice in 1d array.
        self.ref = np.zeros((self.width * self.height), dtype = np.float16) #Identical for recording refractoriness.
        self.Neighbours()

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
    
    def CoupleDel(self):
        '''
        Have dictionary with neighbours. If we take a neighbour away from 1 say 2, need to take 1 from 2 as well
        '''
        keys = self.neighbours.keys()
        x = 0
        y = 0
        new_dic = {i : [] for i in range(len(keys))}
        deleted_dic = {}
        for i in keys:
            neighbours = self.neighbours[i]
            new, deleted =  choose_numbers(neighbours, self.coupling)
            self.neighbours[i] = new
            deleted_dic[i] = deleted
            x += len(deleted)
        
        for i in deleted_dic.keys():
            neighbours = deleted_dic[i]
            for j in neighbours:
                neighbours1 = list(self.neighbours[j])
                if i in neighbours1:
                    index = neighbours1.index(i)
                    neighbours2 = np.delete(neighbours1, index)
                    self.neighbours[j] = neighbours2

        for i in keys:
            neighbours = self.neighbours[i]
            for j in neighbours:
                y += 1
        return y


    def Initialise(self):
        self.index_int = [i*self.width for i in range(0,self.height)] #Left hand side
        #self.hexagon[index_init] = 100 
        self.ref[self.index_int] = 1

    def SigmoidDist(self,charges):
        return 1/(1+np.exp(-self.sig_st*(charges-self.threshold)))

    def ActivationCheck(self):
        index_charged = np.where(self.hexagon > 0)[0]
        p = self.SigmoidDist(self.hexagon[index_charged])
        a = np.random.rand(len(index_charged))
        self.index_act = index_charged[np.where(p>a)[0]]
        self.ref[self.index_act] = 1 #Set sites to activated.
        self.hexagon[index_charged] = 0
        if self.t % 100 == 0:
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
            self.ref[np.where(self.ref >= 1)[0]] += 1
            self.ref[np.where(self.ref == self.ref_per + 2)[0]] = 0
            #self.ref[np.where(self.ref == -1)[0]] = 1

    def RunIt(self):
        self.t = 0
        #self.RefHistory = np.zeros((self.runtime)  * len(self.ref), dtype = np.int16)
        self.RefHistory = np.zeros((500)  * len(self.ref), dtype = np.int16)
        i = 0
        while self.t < self.runtime:
            if self.t == 0:
                self.Initialise()
                self.ActivationCheck()
                self.ChargeProp()
                self.RefHistory[0:len(self.ref)] = self.ref
                i += 1
                self.StateDevelop()
                self.t += self.dt
            elif self.t % 100 == 0:
                self.Initialise()
            self.ActivationCheck()
            self.ChargeProp()
            if i < 500:
                self.RefHistory[i*len(self.ref):(i+1)*len(self.ref)] = self.ref
                i += 1
            else:
                self.RefHistory[0:len(self.ref)] = self.ref
                i = 1
            #self.RefHistory[self.t*len(self.ref):(self.t+1)*len(self.ref)] = self.ref
            self.StateDevelop()
            self.t += self.dt
        np.save('StateData.npy', self.RefHistory)


def main():
    t0 = time.time()
    seed = np.random.randint(0,int(1e7))
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
    lattice.RunIt()
    t1 = time.time()
    print('Runtime = %f s' % (t1-t0))

if __name__ == '__main__':
    #main()
    pass




seed = int(1e6)
np.random.seed(seed)
width = 50
height = 50
runtime = 100000
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
lp.print_stats()
