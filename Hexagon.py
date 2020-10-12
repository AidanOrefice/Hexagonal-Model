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

class HexagonalLattice():
    def __init__(self,height,width):
        self.height = height #Length of hexagonal lattice in units of # of cells.
        self.width = width #Width of hexagonal lattice in units of # of cells.

    def CreateLattice(self):
        self.Hexagon = np.zeros((self.width,self.height)) #Rotate coordinate system by 60 degrees. Still only diamonds
        print(self.Hexagon)

