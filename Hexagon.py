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
    def __init__(self,width,height):
        self.height = height
        self.width = width

        #Ensuring lattice is of the correct dimensions - for toroidal geometry lattice must be even int x even int
        if not(self.width % 2 == 0) or not(self.height % 2 == 0):
            raise ValueError('The lattice must be of dimensions (even integer) x (even integer).')

    def CreateLattice(self):
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


def main():
    lattice = HexagonalLattice(10,10)
    lattice.CreateLattice()
    print(lattice.hexagon)
    lattice.Neighbours()
    print(lattice.neighbours)

if __name__ == '__main__':
    main()
