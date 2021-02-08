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

def InitialLattice(x = 0):
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
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'AF_time','per_%', 'title', 'mean', 'variance', 'in AF?']) #Other columns - need animate? and fname
    df = pd.DataFrame(columns=columns)
    return df

def NormalModesPS():
    df = InitialDF()
    amps = np.linspace(0,0.5,11)
    amps = np.append(amps, [0.75,1,2,5,10,25,50,100,1000,10000,100000])
    offs = np.linspace(0,1,21)  #Same width in each direction.
    A = 1
    runs = 25
    print(A)
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for _ in range(runs):
                lattice = InitialLattice()

                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [A,a,o],
                config['grad_start'], config['grad_end'] )

                run = lattice.RunIt()
                #lattice.Coupling_Sample(A,a,o)
                run[13] = [A,a,o]

                in_AF = lattice.kill

                run.extend([lattice.mean, lattice.var, in_AF]) 
                df.loc[len(df)] = run
    df.to_csv('Normal_Modes_Phase_Space_{}.csv'.format(str(A)))
    return df

def AnimationGrab():
    df = InitialDF()
    offs = np.linspace(0.4,1,13)
    amps = [0]
    A = [1]
    #Will potentially do up to 80 but it wont -- just being systematic. All data will be saved in AF run
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for i in A:
                print('A: ', i)
                lattice = InitialLattice(x = o)

                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [i,a,o],
                config['grad_start'], config['grad_end'] )

                run = lattice.RunIt()
                run[13] = [i,a,o]

                in_AF = lattice.kill
                lattice.Coupling_Sample(i,a,o)


                run.extend([lattice.mean, lattice.var, in_AF]) 
                df.loc[len(df)] = run
    df.to_csv('AnimationRun.csv')
    return df

def Periodicity():
    df = InitialDF()
    amp = 0.2
    off = 0.5
    A = [1]
    runs = 100
    fun = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for i in A:
        print('A:', i)
        for _ in range(runs):
            if _ % 100 == 0:
                print(_)
            for x in fun:
                lattice = InitialLattice(x = i)
                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [i,amp,off],
                config['grad_start'], config['grad_end'])
                run = lattice.RunIt()
                #lattice.Coupling_Sample(i,amp,off)
                #VizTest(i,amp,off,100,100)

                run[13] = [i,amp,off]

                in_AF = lattice.kill #AF_stats(lattice) Did it enter AF
                run.extend([lattice.mean, lattice.var, in_AF]) 
                df.loc[len(df)] = run
    df.to_csv('Prelim_{}.csv'.format(off))
    return df

def bond_counts():
    df = InitialDF()
    offs = [0,0.5,0.75,1]
    runs = 10
    bonds = {i : [] for i in offs}
    for o in offs:
        for _ in range(runs):
            lattice = InitialLattice(x = o)

            lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [1,0,o],
            config['grad_start'], config['grad_end'] )
            bonds[o].append(lattice.number_of_bonds()/ 29800)
    print(bonds)
    '''first = True
    for i in bonds.keys():
        bonds[i] = np.mean(bonds[i]) / np.mean(bonds[1.0])
        if first == True and bonds[i] > 2*np.sin(np.pi/18):
            first = i * 100
    save_dict('bonds_dict', bonds)
    f, ax = plt.subplots()
    ax.bar(range(len(bonds)), list(bonds.values()), align='center', color= 'dimgrey')
    plt.ylabel('Probability of a bond being filled')
    plt.xlabel('Offset * 100')
    plt.hlines(2*np.sin(np.pi/18), 0, 100, linestyles = 'dashed', colors = 'red', label = 'Bond percolation threshold')
    plt.xlim(0,100)
    plt.vlines(first, 0, 1, linestyles = 'dashed', colors = 'green', label = 'Coupling percolation threshold')
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('bonds_bar.png')'''

def main():
    t0 = time.time()
    df = Periodicity()
    '''for i in range(len(df)):
        Animate(str(df['title'][i]),str(df['FullStateSave'][i]), df['location_2'][i], df['location_3'][i], df['location_4'][i], df['normal_modes_config'][i])'''
    
    t1 = time.time()
    print(t1-t0)

if __name__ == '__main__':
    main()




'''
How to use Line Profiler on a method of a class.
lp = LineProfiler()
lp_wrapper = lp(class.method)
lp_wrapper()
lp.print_stats()
'''