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

def index_to_xy(index):
    row = np.floor(index / 100)
    y = row - row*(1-(np.sqrt(3)/2)) #fix.
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * 100)
    else:
        x = index - (row * 100) + 0.5
    return (x,y)

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
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'location_err', 'AF_time', 'Hamming_dis_arr', 'Ham_dis_AF', 'Ham_dis_meanx', 'per_%', 'title', 'mean', 'variance', 'in AF?', 'multiplier']) #Other columns - need animate? and fname
    df = pd.DataFrame(columns=columns)
    return df

def InitialDF_Ham():
    columns = list(config.keys())
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'location_err', 'AF_time', 'Hamming_dis_arr', 'Ham_dis_AF', 'Ham_dis_meanx', 'per_%', 'title', 'mean', 'variance', 'in AF?', 'multiplier']) #Other columns - need animate? and fname
    columns.extend(['Ham_dis_beats_avg_over'])
    columns.extend(['Ham_dis_fib_beat?'])
    columns.extend([str(i) for i in range(200)])
    df = pd.DataFrame(columns=columns)
    return df

def NormalModesPS():
    df = InitialDF()
    amps = np.linspace(0,0.5,11)
    #amps = np.append(amps, [0.75,1,2,5,10,25,50,100,1000,10000,100000])
    offs = np.linspace(0,1,21)  #Same width in each direction.
    A = 2
    runs = 25
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
    off = 0.8 #0.4,0.5,0.6,0.7,0.
    a = [1]
    multi = {0.4 : 0.015, 0.5 : 0.04, 0.6 : 0.08, 0.7 : 0.8, 0.8 : 1.42}
    print(multi[off])
    runs = 10000
    for A in a:
        print('Off', off)
        for _ in range(runs):
            if _ % 100 == 0:
                print(_)

            lattice = InitialLattice(x = multi[off])
            lattice.CouplingMethod([A,amp,off])
            run = lattice.RunIt()

            run[8] = [A,amp,off]
            in_AF = lattice.kill #AF_stats(lattice) Did it enter AF
            run.extend([lattice.mean, lattice.var, in_AF, multi[off]]) 
            df.loc[len(df)] = run
    df.to_csv('FailureMultiplierData_{}_full.csv'.format(off))
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

def Hamming_distance(time_data):
    activated_sites = np.where(time_data == 1)[0]
    activated_sites_x = [index_to_xy(i)[0] for i in activated_sites]
    if len(activated_sites) > 0:
        x_mean = np.mean(activated_sites_x)
        Ham_dis = np.sum((activated_sites_x-x_mean)**2)/len(activated_sites_x)
        return np.sqrt(Ham_dis)
    else:
        return 0

def Ham_dis_inves(lattice,run):
    '''
    Important things:
    - Do not calculate on asymptotic beats
    - Calculate separately on AFIB beats
    - Save as dataframe with one row of 1-199
    '''
    run_fib = run.copy()
    run_no_fib = run.copy()
    if lattice.kill:
        AF_beat = int(np.floor((lattice.AF_time[1]-100) / lattice.pacing_period))
        Hamming_dis_fib = [False for i in range(200)]
        for j in range(lattice.AF_time[1]-100 - AF_beat * 200):
            time = AF_beat*200 + j
            beat_data = lattice.RefHistory[int(time*10000):int((time+1)*10000)]
            Hamming_dis_fib[j] = Hamming_distance(beat_data)
        run_fib.extend([1])
        run_fib.extend([True])
        run_fib.extend(Hamming_dis_fib)
    else:
        AF_beat = 50
        run_fib.extend([0])
        run_fib.extend([False])
        Hamming_dis_fib = [False for i in range(200)]
        run_fib.extend(Hamming_dis_fib)
    Hamming_dis_non_fib = {i : [] for i in range(200)}
    for i in range(AF_beat):
        if lattice.AF_bool[i][1] == False:
            for j in range(200):
                time = i*200 + j
                beat_data = lattice.RefHistory[int(time*10000):int((time+1)*10000)]
                Hamming_dis_non_fib[j].append(Hamming_distance(beat_data))
    if AF_beat > 0:
        run_no_fib.extend([AF_beat])
        Hamming_dis_non_fib_mean = [np.mean(Hamming_dis_non_fib[i]) for i in range(200)]
    else:
        run_no_fib.extend([0])
        Hamming_dis_non_fib_mean = [False for i in range(200)]
    run_no_fib.extend([False])
    run_no_fib.extend(Hamming_dis_non_fib_mean)
    return run_fib, run_no_fib

def Hamming_dis_graph_data():
    df = InitialDF_Ham()
    periodicity = [1,2,3,5,7,10,13,16,20]
    runs = 6
    offs = np.linspace(0.4,1,13)
    amps = np.linspace(0,0.5,11)
    print(offs,amps)
    for off in offs:
        print('off : {}'.format(off))
        for amp in amps:
            print('amp : {}'.format(amp))
            for a in periodicity:
                for _ in range(runs):
                    lattice = InitialLattice(x = 1)
                    lattice.CouplingMethod([a,amp,off])
                    run = lattice.RunIt()

                    run[8] = [a,amp,off]
                    in_AF = lattice.kill #AF_stats(lattice) Did it enter AF
                    run.extend([lattice.mean, lattice.var, in_AF, 1]) 
                    #Some Hamming distance investigation:
                    run_fib, run_no_fib = Ham_dis_inves(lattice,run)
                    df.loc[len(df)] = run_fib
                    df.loc[len(df)] = run_no_fib
    df.to_csv('Ham_dis_run_fib_PS_{}.csv'.format(6))
    return df

def Toy_Anim():
    #Need to make sure config is set properly.
    #Create a toy animation with hamming distance overlaid
    off = 1
    amp = 0.2
    A = 1
    lattice = InitialLattice()
    lattice.CouplingMethod([A,amp,off])
    run = lattice.RunIt()
    #Animate(str(run[-1]), 'full', 0, 0, 0, '[%s, %s, %s]' % (str(A), str(amp), str(off)), True)

def Ham_dis_Max():
    a = 10
    off = 0.8
    amp = 0.4
    runs = range(10)
    for i in runs:
        print(i)
        lattice = InitialLattice(x = 1)
        lattice.CouplingMethod([a,amp,off])
        run = lattice.RunIt()
        if lattice.kill:
            AF_beat = int(np.floor((lattice.AF_time[1]-100) / lattice.pacing_period))
            print(AF_beat)
            if AF_beat < 10:
                if AF_beat > 2:
                    Ham_dis_dict = {i : [] for i in range(AF_beat+1)}
                    times = [i for i in range(200)]
                    af_time = (lattice.AF_time[1] - 100) - AF_beat * 200
                    for i in range(AF_beat + 1):
                        for j in range(200):
                            time = i*200 + j
                            beat_data = lattice.RefHistory[int(time*10000):int((time+1)*10000)]
                            Ham_dis_dict[i].append(Hamming_distance(beat_data))
                        if i < 1:
                            plt.plot(times, Ham_dis_dict[i], color = 'black', label = 'Normal beats')
                        elif i < AF_beat:
                            plt.plot(times, Ham_dis_dict[i], color = 'black')
                        else:
                            plt.plot(times[:af_time + 2], Ham_dis_dict[i][:af_time + 2], color = 'red', label = 'Fibrillation beat')
                    plt.vlines((lattice.new_time) - AF_beat * 200, 0, max(Ham_dis_dict[i][:af_time + 2]), color = 'blue', ls = '--', label = 'Fibrillation initiation')
                    plt.ylabel('Hamming Distance')
                    plt.xlabel('Time after beat initiation')
                    plt.legend()
                    plt.savefig('fib_vs_non_fib_10_{}.png'.format(lattice.seed))

def Ham_dis_Max1():
    a = 5
    off = 0.8
    amp = 0.3
    runs = range(10)
    for i in runs:
        print(i)
        lattice = InitialLattice(x = 1)
        lattice.CouplingMethod([a,amp,off])
        run = lattice.RunIt()
        if lattice.kill:
            AF_beat = int(np.floor((lattice.AF_time[1]-100) / lattice.pacing_period))
            print(AF_beat)
            if AF_beat < 51:
                if AF_beat > 2:
                    Ham_dis_dict = {i : [] for i in range(AF_beat+1)}
                    times = [i for i in range(200)]
                    af_time = (lattice.AF_time[1] - 100) - AF_beat * 200
                    for i in range(AF_beat + 1):
                        for j in range(200):
                            time = i*200 + j
                            beat_data = lattice.RefHistory[int(time*10000):int((time+1)*10000)]
                            Ham_dis_dict[i].append(Hamming_distance(beat_data))
                    mean_ham_t = np.asarray([np.mean([Ham_dis_dict[k][j] for k in range(AF_beat)]) for j in range(200)])
                    std_ham_t = np.asarray([np.std([Ham_dis_dict[k][j] for k in range(AF_beat)]) for j in range(200)])
                    plt.plot(times, mean_ham_t, color = 'black', label = 'Average of {} normal beats'.format(AF_beat - 1))
                    plt.fill_between(times, mean_ham_t + std_ham_t, mean_ham_t - std_ham_t, color = 'gray', label = 'Standard deviation')
                    plt.plot(times[:af_time + 2], Ham_dis_dict[i][:af_time + 2], color = 'red', label = 'Fibrillation beat')
                    plt.vlines((af_time - 12), 0, max(Ham_dis_dict[i][:af_time + 2]), color = 'blue', ls = '--', label = 'Fibrillation initiation')
                    plt.ylabel('Hamming Distance')
                    plt.xlabel('Time after beat initiation')
                    plt.legend()
                    plt.savefig('fib_vs_non_fib_5_{}.png'.format(lattice.seed))
                    plt.close()

                    plt.plot(times[:af_time], Ham_dis_dict[i][:af_time] - mean_ham_t[:af_time], color = 'red')
                    plt.vlines((af_time - 12), 0, max(Ham_dis_dict[i][:af_time + 2]), color = 'blue', ls = '--', label = 'Fibrillation initiation')
                    plt.savefig('fib_vs_non_fib_5_minus_{}.png'.format(lattice.seed))
                    plt.close()




def main():
    dat = pd.read_csv('FailureMultiplierData_0.7_full.csv')
    dat = dat[dat['in AF?']]
    dat = dat[dat['location_err']]
    times = sorted([(int(i.split(',')[1].split(')')[0]) - 100)%200 for i in dat['AF_time']])
    med_t, mean_t = np.median(times), np.mean(times)
    print('Median:', med_t)
    print('Mean:', mean_t)
    #df = Toy_Anim
    '''for i in range(len(df)):
        Animate(str(df['title'][i]),str(df['FullStateSave'][i]), df['location_2'][i], df['location_3'][i], df['location_4'][i], df['normal_modes_config'][i])'''

            


if __name__ == '__main__':
    t0 = time.time()
    NormalModesPS()
    t1 = time.time()
    print(t1-t0)



'''
How to use Line Profiler on a method of a class.
lp = LineProfiler()
lp_wrapper = lp(class.method)
lp_wrapper()
lp.print_stats()
'''