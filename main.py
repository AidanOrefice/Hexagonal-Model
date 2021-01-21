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
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'AF_time','per_%', 'title', 'in AF?']) #Other columns - need animate? and fname
    df = pd.DataFrame(columns=columns)
    return df

def AF_stats(lattice):
    index = np.where(lattice.AF > 55)[0]
    thing = [list(map(itemgetter(1), g)) for tk, g in groupby(enumerate(index), lambda ix : ix[0] - ix[1])]
    thing = [i for i in thing if len(i) > 10]
    len_thing = 0
    if len(thing) > 0:
        in_AF = True
        for x in thing:
            len_thing += len(x)
    else:
        in_AF = False

    fraction_in_AF = len_thing/config['runtime']

    return in_AF, fraction_in_AF

def NormalModesPS():
    df = InitialDF()
    amps = np.linspace(0,0.5,26)
    amps = np.append(amps, [0.75,1,2,5,10])
    offs = np.linspace(0.2,0.8,31)  #Same width in each direction.
    Ax, Ay = 10,3
    runs = 1
    for o in offs:
        print('Offset:', o)
        for a in amps:
            print('Amplitude:', a)
            for _ in range(runs):
                lattice = InitialLattice()

                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [Ax,Ay,a,o],
                config['grad_start'], config['grad_end'] )

                run = lattice.RunIt()
                lattice.Coupling_Sample(Ax,Ay,a,o)
                run[13] = [Ax,Ay,a,o]

                in_AF = lattice.kill#AF_stats(lattice) Did it enter AF

                run.extend([in_AF]) 
                df.loc[len(df)] = run
    df.to_csv('Prelim.csv')
    return df

def Periodicity():
    df = InitialDF()
    amp = 0.2
    off = 0.75
    Ax = [1,3,5,8,10,20]
    Ay = [1,3,5,8,10]
    runs = 10
    for i1 in Ax:
        print('Ax:', i1)
        for i2 in Ay:
            print('Ay:', i2)
            for _ in range(runs):
                lattice = InitialLattice()
                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [i1,i2,amp,off],
                config['grad_start'], config['grad_end'] )
                run = lattice.RunIt()
                if _ == 1:
                    lattice.Coupling_Sample(i1,i2,amp,off)
                    VizTest(i1,i2,amp,off,80,140)

                run[13] = [i1,i2,amp,off]

                in_AF = lattice.kill #AF_stats(lattice) Did it enter AF

                run.extend([in_AF]) 
                df.loc[len(df)] = run
    df.to_csv('PeriodicityInvestigation.csv')
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