"""
Authors: Daniel Loughran and Aidan Orefice
Script to initialise the hexagonal lattice

Essentially will be a network with each site having neighbours - apart from boundary cases.

Need to be able to randomly remove bonds between certain sites and track which sites are bonded.

Initialise it with some cell size dimensions i.e. 50 x 100 cells

Think of a way to use an array - for efficiency, dont really want to be using networkx package.

3 element coordinate system - hexagonal root
"""

import numpy as np
import math

class HexagonalLattice():
    def __init__(self,height,width):
        self.height = height #Length of hexagonal lattice in units of # of cells.
        self.width = width #Width of hexagonal lattice in units of # of cells.

    def CreateLattice(self):
        self.hexagon = np.zeros((self.width * self.height), dtype = np.float16)
        self.ref = np.zeros((self.width * self.height), dtype = np.float16)

    def Neighbours(self):
        #Neighbours with a taurus, without a taurus change is pretty easy
        self.neighbours = {}
        for i in range(self.height):
            if i == 0: #First row always even
                for j in range(0, self.width):
                    if j == 0:
                        self.neighbours[0] = np.asarray([self.width, 1, (self.width * 2) - 1, self.width - 1, self.width * (self.height -1), self.width * self.height - 1])
                    elif j == self.width - 1:
                        self.neighbours[j] = np.asarray([j - 1, j + self.width - 1, j + self.width, 0, self.width * self.height - 1, self.width * self.height - 2])
                    else:
                        self.neighbours[j] = np.asarray([j - 1, j + 1, j + self.width - 1, j + self.width, self.width * (self.height - 1) + j - 1, self.width * (self.height - 1) + j])
            elif i == self.height - 1: #Last row always odd
                for j in range(0, self.width):
                    index = i * self.width + j
                    if j == 0:
                        self.neighbours[index] = np.asarray([index - self.width, index - self.width + 1, index + 1, self.width * self.height - 1, 0, 1])
                    elif j == self.width - 1:
                        self.neighbours[index] = np.asarray([index - (self.width * 2) + 1, index - self.width, index - 1, index - self.width + 1, 0, self.width - 1])
                    else:
                        self.neighbours[index] = np.asarray([index - self.width, index - self.width + 1, index - 1, index + 1, j, j + 1])
            else:
                if_even = i % 2 == 0 #False is odd True is even
                for j in range(0, self.width):
                    index = i * self.width + j
                    if j == 0:
                        if if_even:
                            self.neighbours[index] = np.asarray([index - 1, index + self.width - 1, index + self.width, index - self.width, index + 1, index + (self.width * 2) - 1])
                        else:
                            self.neighbours[index] = np.asarray([index + 1, index + self.width - 1, index + self.width + 1, index + self.width, index - self.width + 1, index - self.width])
                    elif j == self.width - 1:
                        if if_even:
                            self.neighbours[index] = np.asarray([index - self.width + 1, index - 1, index + self.width - 1, index + self.width, index - self.width - 1, index - self.width])
                        else:
                            self.neighbours[index] = np.asarray([index - 1, index - self.width + 1, index + self.width, index - self.width, index + 1, index - (self.width * 2) + 1])
                    else:
                        if if_even:
                            self.neighbours[index] = np.asarray([index + 1, index - 1, index + self.width - 1, index + self.width, index - self.width - 1, index - self.width])
                        else:
                            self.neighbours[index] = np.asarray([index + 1, index - 1, index + self.width + 1, index + self.width, index - self.width + 1, index - self.width])

def main():
    lattice = HexagonalLattice(4,4)
    lattice.CreateLattice()
    print(lattice.hexagon)
    lattice.Neighbours()
    print(lattice.neighbours)

if __name__ == '__main__':
    main()
