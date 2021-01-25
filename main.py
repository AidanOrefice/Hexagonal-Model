from Animation import *
from Hexagon import *
from configuration import *
from CouplingViz import *
import time 

def InitialLattice():
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
    return lattice

def InitialDF():
    columns = list(config.keys())
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'AF_time','per_%', 'title', 'mean', 'variance', 'in AF?']) #Other columns - need animate? and fname
    df = pd.DataFrame(columns=columns)
    return df

def NormalModesPS():
    df = InitialDF()
    amps = np.linspace(0,0.5,26)
    amps = np.append(amps, [0.75,1,2,5,10])
    offs = np.linspace(0.2,0.8,31)  #Same width in each direction.
    A = 5
    runs = 1
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for _ in range(runs):
                lattice = InitialLattice()

                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [A,a,o],
                config['grad_start'], config['grad_end'] )

                run = lattice.RunIt()
                lattice.Coupling_Sample(A,a,o)
                run[13] = [A,a,o]

                in_AF = lattice.kill#AF_stats(lattice) Did it enter AF

                run.extend([lattice.mean, lattice.var, in_AF]) 
                df.loc[len(df)] = run
    df.to_csv('Prelim.csv')
    return df

def Periodicity():
    df = InitialDF()
    amp = 0.2
    off = 0.7
    A = [3]
    runs = 1
    for i in A:
        print('A:', i)
        for _ in range(runs):
            lattice = InitialLattice()
            lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [i,amp,off],
            config['grad_start'], config['grad_end'] )
            run = lattice.RunIt()
            lattice.Coupling_Sample(i,amp,off)
            VizTest(i,amp,off,100,100)

            run[13] = [i,amp,off]

            in_AF = lattice.kill #AF_stats(lattice) Did it enter AF
            run.extend([lattice.mean, lattice.var, in_AF]) 
            df.loc[len(df)] = run
    df.to_csv('Prelim.csv')
    return df

def main():
    t0 = time.time()

    df = Periodicity()
    '''
    for i in range(len(df)):
        Animate(str(df['title'][i]),str(df['FullStateSave'][i]), df['location_2'][i], df['location_3'][i], df['location_4'][i])'''
    
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