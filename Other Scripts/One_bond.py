from Hexagon import HexagonalLattice
from time import time
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt

def Test():
    seed = np.random.randint(0,int(1e7))
    np.random.seed(seed)
    lattice = HexagonalLattice(4,4,1,0.1,20, 1)
    lattice.CreateLattice()
    lattice.Remove_random_bonds(1)
    print(lattice.neighbours)
    lattice.Remove_random_bonds(1)
    print(lattice.neighbours)
    lattice.Remove_random_bonds(1)
    print(lattice.neighbours)

def Main_random_seed():
    t1 = time()
    seed = np.random.randint(0,int(1e7))
    np.random.seed(seed)
    thresholds = [0.4,0.5]
    for j in thresholds:
        df = pd.DataFrame(columns = ['settings', 'in_AF', 'coupling', '%_time_AF'])
        print(j)
        for i in range(6000):
            coupling_per = 1 - ((i + 1) / 7400)
            lattice = HexagonalLattice(50,50,1000,j,25)
            lattice.CreateLattice()
            lattice.Remove_random_bonds(i+1)
            lattice.RunIt()
            index = np.where(lattice.AF > 55)[0]
            thing = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(index), lambda ix : ix[0] - ix[1])]
            thing = [i for i in thing if len(i) > 10]
            len_thing = 0
            if len(thing) > 0:
                in_AF = True
                for x in thing:
                    len_thing += len(x)
            else:
                in_AF = False
            df1 = pd.DataFrame([[[50,50,1000,j,25], in_AF, coupling_per, len_thing/1000]], columns = ['settings', 'in_AF', 'coupling', '%_time_AF'])
            df = df.append(df1, ignore_index = True)
        df.to_csv('One_bond_{}.csv'.format(j))
        t2 = time()
        print(t2-t1)

def Main_set_seed():
    t1 = time()
    seed = np.random.randint(0,int(1e7))
    np.random.seed(seed)
    df = pd.DataFrame(columns = ['settings', 'in_AF', 'coupling', '%_time_AF'])
    #thresholds = range(0.1,0.5,0.1)
    thresholds = [0.3]
    for j in thresholds:
        lattice = HexagonalLattice(50,50,1000,j,25)
        lattice.CreateLattice()
        for i in range(6000):
            coupling_per = 1 - ((i + 1) / 7400)
            lattice.Remove_random_bonds(1)
            lattice.RunIt()
            index = np.where(lattice.AF > 55)[0]
            thing = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(index), lambda ix : ix[0] - ix[1])]
            thing = [i for i in thing if len(i) > 10]
            len_thing = 0
            if len(thing) > 0:
                in_AF = True
                for x in thing:
                    len_thing += len(x)
            else:
                in_AF = False
            print(len_thing)
            df1 = pd.DataFrame([[[50,50,1000,j,25], in_AF, coupling_per, len_thing/10000]], columns = ['settings', 'in_AF', 'coupling', '%_time_AF'])
            df = df.append(df1, ignore_index = True)
            lattice.CreateLatticeNoNeighbours()
        df.to_csv('One_bond_re_{}.csv'.format(j))
        t2 = time()
        print(t2-t1)

def plot(file):
    f, ax = plt.subplots()
    thing = pd.read_csv(file)
    thing['in_AF1'] = thing['in_AF'].astype(int)
    thing['in_AF2'] = 0
    for i in range(len(thing)):
        if i < 50:
            thing.loc[i,'in_AF2'] = np.average(thing['in_AF1'][0:i+51])
        elif i > len(thing) - 50:
            thing.loc[i,'in_AF2'] = np.average(thing['in_AF1'][i-50:len(thing)-1])
        else:
            thing.loc[i,'in_AF2'] = np.average(thing['in_AF1'][i-50:i+51])
    thing.plot(x = 'coupling', y = 'in_AF2', ax = ax, ls = '-', legend = False)
    plt.ylabel('Averaged in AF')
    plt.savefig(file + str('.png'))

def plot_PER(file):
    f, ax = plt.subplots()
    thing = pd.read_csv(file)
    for i in range(len(thing)):
        if i < 50:
            thing.loc[i,'in_AF2'] = np.average(thing['%_time_AF'][0:i+51])
        elif i > len(thing) - 50:
            thing.loc[i,'in_AF2'] = np.average(thing['%_time_AF'][i-50:len(thing)-1])
        else:
            thing.loc[i,'in_AF2'] = np.average(thing['%_time_AF'][i-50:i+51])
    thing.plot(x = 'coupling', y = 'in_AF2', ax = ax, ls = '-', legend = False)
    plt.ylabel('Averaged % time spent in AF')
    plt.savefig(file + str('_PER') + str('.png'))


    
if __name__ == '__main__':
    #'One_bond_0.1.csv','One_bond_0.2.csv','One_bond_0.3.csv','One_bond_0.4.csv','One_bond_0.5.csv',
    for i in ['One_bond_re_0.3.csv']:
        plot(i)
        plot_PER(i)

'''
each point we want average of 10 points either side'''
