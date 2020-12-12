from Animation import *
from Hexagon import *
from configuration import *
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
    columns.extend(['seed','Location_1','Time_1','Location_2','Time_2', 'Per_%', 'title', 'in AF?', '%time in AF',]) #Other columns - need animate? and fname
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
    amps = [0.5]#np.linspace(0,0.5,21)
    means = [0.3]#np.linspace(0,1,21)
    print(means)
    print(amps)
    for k in means:
        for i in amps:
            for _ in range(1):
                lattice = InitialLattice()

                lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], [i,k],
                config['grad_start'], config['grad_end'] )

                run = lattice.RunIt()
                run[13] = [0.25,1,i,k]

                in_AF, fraction_in_AF = AF_stats(lattice)

                run.extend([in_AF, fraction_in_AF]) 
                df.loc[len(df)] = run
                print([i[0] for i in lattice.AF_bool if i[1] == True])
    df.to_csv('Prelim.csv')
    return df

def main():
    t0 = time.time()

    lattice = InitialLattice()
    lattice.CouplingMethod(config['constant'], config['gradient'], config['normal_modes'], config['normal_modes_config'][2:],
     config['grad_start'], config['grad_end'] )
    lattice.RunIt()
    
    t1 = time.time()
    print(t1-t0)

if __name__ == '__main__':
    df = NormalModesPS()
    Animate(df.loc[0,'title'], 'transition', int(df.loc[0,'Location_1']), int(df.loc[0,'Location_2']))




'''
How to use Line Profiler on a method of a class.
lp = LineProfiler()
lp_wrapper = lp(class.method)
lp_wrapper()
lp.print_stats()
'''