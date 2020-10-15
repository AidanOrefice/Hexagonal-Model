"""
Start Date: 12/10/2020
Authors: Daniel Loughran and Aidan Orefice
Script to initialise the hexagonal lattice

Create a simple animation to highlight the basic mechanics of lattice are working. 
"""

import numpy as np

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
    def __init__(self,width,height, runtime):
        self.height = height
        self.width = width
        self.dt = 1 #Discrete time width of lattice.
        self.threshold = 1
        self.runtime = runtime #Total time we want to run for.

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
                        self.neighbours[0] = np.asarray([self.width, 1, (self.width * 2) - 1, self.width - 1,
                         self.width * (self.height -1), self.width * self.height - 1])
                    elif j == self.width - 1: #bottom right corner
                        self.neighbours[j] = np.asarray([j - 1, j + self.width - 1, j + self.width, 0,
                         self.width * self.height - 1, self.width * self.height - 2])
                    else:
                        self.neighbours[j] = np.asarray([j - 1, j + 1, j + self.width - 1, j + self.width,
                         self.width * (self.height - 1) + j - 1, self.width * (self.height - 1) + j])
            elif i == self.height - 1: #Last row always odd
                for j in range(0, self.width):
                    index = i * self.width + j
                    if j == 0: #top left corner
                        self.neighbours[index] = np.asarray([index - self.width, index - self.width + 1,
                         index + 1, self.width * self.height - 1, 0, 1])
                    elif j == self.width - 1: #top right corner
                        self.neighbours[index] = np.asarray([index - (self.width * 2) + 1, index - self.width,
                         index - 1, index - self.width + 1, 0, self.width - 1])
                    else:
                        self.neighbours[index] = np.asarray([index - self.width, index - self.width + 1,
                         index - 1, index + 1, j, j + 1])
            else: #All intermediate rows.
                if_even = i % 2 == 0 #False is odd True is even
                for j in range(0, self.width):
                    index = i * self.width + j
                    if j == 0: #left-most column
                        if if_even:
                            self.neighbours[index] = np.asarray([index - 1, index + self.width - 1,
                             index + self.width, index - self.width, index + 1, index + (self.width * 2) - 1])
                        else:
                            self.neighbours[index] = np.asarray([index + 1, index + self.width - 1,
                             index + self.width + 1, index + self.width, index - self.width + 1, index - self.width])
                    elif j == self.width - 1: #right-most column
                        if if_even:
                            self.neighbours[index] = np.asarray([index - self.width + 1, index - 1,
                             index + self.width - 1, index + self.width, index - self.width - 1, index - self.width])
                        else:
                            self.neighbours[index] = np.asarray([index - 1, index - self.width + 1,
                             index + self.width, index - self.width, index + 1, index - (self.width * 2) + 1])
                    else: #All non-edge sites.
                        if if_even:
                            self.neighbours[index] = np.asarray([index + 1, index - 1, index + self.width - 1,
                             index + self.width, index - self.width - 1, index - self.width])
                        else:
                            self.neighbours[index] = np.asarray([index + 1, index - 1, index + self.width + 1,
                             index + self.width, index - self.width + 1, index - self.width])

    def Initialise(self):
        index_init = [i*self.width for i in range(0,self.height)]
        self.hexagon[index_init] = 3*self.threshold
        self.ref[index_init] = 1
        self.t = self.dt

    def SigmoidDist(self,charges):
        return 1/(1+np.exp(-10*(charges-self.threshold)))

    def ActivationCheck(self):
        index_rest = np.where(self.ref == 0)[0]
        p = self.SigmoidDist(self.hexagon[index_rest])
        a= np.random.rand(len(index_rest))
        b = index_rest[np.where(p>a)[0]]
        self.ref[b] = -1 #pseudo-state: just activated


    def ChargeProp(self):
        RefHistory = a = np.zeros(shape=(self.runtime,self.width*self.height))
        while self.t <= self.runtime:
            RefHistory[self.t-1] = self.ref
            index_charged = np.where(self.ref == 1)[0] #sites that are activated - need to spread their charge
            for key in index_charged:
                neighbours = self.neighbours[key]
                avail_neighbours = []
                for i in neighbours:
                    if self.ref[i] == 0:
                        avail_neighbours.append(i)  #retrieved their resting neighbours
                amplitude = self.hexagon[key]/len(avail_neighbours)
                self.hexagon[avail_neighbours] += amplitude #Spreading out the charge
                self.hexagon[key] = 0

            #Checking which states can be activated.
            self.ActivationCheck()

            self.ref[np.where(self.ref >= 1)[0]] += 1
            self.ref[np.where(self.ref == 5)[0]] = 0
            self.ref[np.where(self.ref == -1)[0]] = 1

            print(self.t)
            self.t += self.dt
        np.save('StateData.npy', RefHistory)


        


def main():
    lattice = HexagonalLattice(50,50,250)
    lattice.CreateLattice()
    lattice.Initialise()
    lattice.ChargeProp()

if __name__ == '__main__':
    main()
