from Animation import *
from Hexagon import *
from configuration import *
from CouplingViz import *
import time 
import pickle


def save_dict(fname, a):
    with open('{}.pickle'.format(fname), 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

def open_dict(fname):
    with open('{}.pickle'.format(fname), 'rb') as handle:
        b = pickle.load(handle)

def InitialLattice(x = 1):
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
        config['set_seed'],
        multiplier = x)

    lattice.CreateLattice()
    return lattice

def InitialDF():
    columns = list(config.keys())
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'location_err', 'AF_time', 'Hamming_dis_arr', 'per_%', 'title', 'mean', 'variance', 'in AF?', 'multiplier']) #Other columns - need animate? and fname
    df = pd.DataFrame(columns=columns)
    return df

def NormalModesPS():
    df = InitialDF()
    amps = np.linspace(0,0.5,11)
    amps = np.append(amps, [0.75,1,2,5,10,25,50,100,1000,10000,100000])
    offs = np.linspace(0,1,21)  #Same width in each direction.
    A = 20
    runs = 15
    print(A)
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for _ in range(runs):
                lattice = InitialLattice()

                lattice.CouplingMethod([A,a,o])
                run = lattice.RunIt()
                #lattice.Coupling_Sample(A,a,o)
                run[8] = [A,a,o]

                in_AF = lattice.kill

                run.extend([lattice.mean, lattice.var, in_AF, 1])
                df.loc[len(df)] = run
    df.to_csv('Normal_Modes_Phase_Space_Ham_dis_{}.csv'.format(str(A)))
    return df

def PercolationGrab():
    df = InitialDF()
    amps = np.linspace(0,0.5,11)
    offs = np.linspace(0.35,0.75,9)
    A = 20
    multi = 1
    runs = 6
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for _ in range(runs):
                lattice = InitialLattice(x = multi)

                lattice.CouplingMethod([A,a,o])
                run = lattice.RunIt()
                #lattice.Coupling_Sample(A,a,o)
                run[8] = [A,a,o]
                in_AF = lattice.kill
                run.extend([lattice.mean, lattice.var, in_AF, multi]) 
                df.loc[len(df)] = run
    df.to_csv('PercolationData_{}.csv'.format(str(A)))
    return df


def AnimationGrab():
    df = InitialDF()
    offs = [0.65]
    amps = [0]
    A = [1]
    #Will potentially do up to 80 but it wont -- just being systematic. All data will be saved in AF run
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for i in A:
                print('A: ', i)
                lattice = InitialLattice()

                lattice.CouplingMethod([i,a,o])

                run = lattice.RunIt()
                run[8] = [i,a,o]

                in_AF = lattice.kill
                lattice.Coupling_Sample(i,a,o)


                run.extend([lattice.mean, lattice.var, in_AF,1]) 
                df.loc[len(df)] = run
    df.to_csv('AnimationRun.csv')
    return df

def Periodicity(): #Ensure Config is set up properly
    df = InitialDF()
    amp = 0.2
    off = 0.75
    A = [1] #0,1,3,5,10,20
    multi = 0.04
    print(multi)
    runs = 10000
    for i in A:
        print('multi', i)
        for _ in range(runs):
            if _ % 10 == 0:
                print(_)

            lattice = InitialLattice(x = multi)
            lattice.CouplingMethod([i,amp,off])
            run = lattice.RunIt()

            run[8] = [i,amp,off]
            in_AF = lattice.kill #AF_stats(lattice) Did it enter AF
            run.extend([lattice.mean, lattice.var, in_AF, multi]) 
            df.loc[len(df)] = run
    df.to_csv('FailureMultiplierData__0.75_{}.csv'.format(i))
    return df

def bond_counts(load = True):
    t0 = time.time()
    offs = np.linspace(0,1,201)
    runs = 10
    if not load:
        bonds = {i : [] for i in offs}
        for o in offs:
            print(o)
            for _ in range(runs):
                lattice = InitialLattice()
                lattice.CouplingMethod([1,0,o])
                bonds[o].append(lattice.number_of_bonds()/ 29800)
    else:
        pickle_in = open("bonds_dict.pickle","rb")
        bonds = pickle.load(pickle_in)
    
    first = True
    for i in bonds.keys():
        bonds[i] = np.mean(bonds[i]) / np.mean(bonds[1.0])
        if first == True and bonds[i] > 2*np.sin(np.pi/18):
            first = i * 100
            print(first)
    if not load:
        save_dict('bonds_dict', bonds)

    new_bonds = []
    #for i in range(len(offs))
    f, ax = plt.subplots()
    ax.bar(100*offs, list(bonds.values()), align='center', color= 'dimgrey')
    ax.plot(100*offs, list(bonds.values()), linestyle = '--', color = 'blue', label = 'Linear Fit')
    plt.ylabel('Probability of a bond being filled')
    plt.xlabel('Offset * 100')
    plt.hlines(2*np.sin(np.pi/18), 0, 100, linestyles = 'dashed', colors = 'red', label = 'Bond percolation threshold')
    plt.xlim(0,100)
    plt.vlines(first, 0, 1, linestyles = 'dashed', colors = 'green', label = 'Coupling percolation threshold')
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('bonds_bar.png')
    t1 = time.time()
    print(t1-t0)

def loc_dis_test(a):
    print('A = ' + str(a))
    lattice = InitialLattice(x = 1)
    lattice.CouplingMethod([a,0.3,0.75])
    run = lattice.RunIt()
    if lattice.kill:
        for i in range(50):
            print(lattice.Hamming_distance(lattice.AF_time[1]-100-i), i)

def Hamming_Dis_graph_data():
    df = InitialDF()
    periodicity = [i for i in range(1,21)]
    runs = 3
    off = 0.8
    amps = [0.2+i*0.1 for i in range(0,4)]
    for amp in amps:
        print(amps)
        for a in periodicity:
            print(a)
            for _ in range(runs):
                lattice = InitialLattice(x = 1)
                lattice.CouplingMethod([a,amp,off])
                run = lattice.RunIt()

                run[8] = [a,amp,off]
                in_AF = lattice.kill #AF_stats(lattice) Did it enter AF
                run.extend([lattice.mean, lattice.var, in_AF, 1]) 
                df.loc[len(df)] = run
    df.to_csv('Ham_dis_run.csv')
    return df

def main():
    df = NormalModesPS()
    '''for i in range(len(df)):
        Animate(str(df['title'][i]),str(df['FullStateSave'][i]), df['location_2'][i], df['location_3'][i], df['location_4'][i], df['normal_modes_config'][i])'''

            


if __name__ == '__main__':
    t0 = time.time()
    Hamming_Dis_graph_data()
    t1 = time.time()
    print(t1-t0)



'''
How to use Line Profiler on a method of a class.
lp = LineProfiler()
lp_wrapper = lp(class.method)
lp_wrapper()
lp.print_stats()
'''