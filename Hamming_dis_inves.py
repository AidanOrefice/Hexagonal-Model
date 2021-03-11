import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from Animation import *
from Hexagon import *
from configuration import *
from CouplingViz import *
import pickle

def index_to_xy(index):
    row = np.floor(index / 100)
    y = row - row*(1-(np.sqrt(3)/2)) #fix.
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * 100)
    else:
        x = index - (row * 100) + 0.5
    return (x,y)

def hamming_distance(time_data):
    '''Takes in one beat worth of data and calculates a Hamming distance'''
    activated_sites = np.where(time_data == 1)[0]
    activated_sites_x = [index_to_xy(i)[0] for i in activated_sites]
    if len(activated_sites) > 0:
        x_mean = np.mean(activated_sites_x)
        Ham_dis = np.sum((activated_sites_x-x_mean)**2)
        return np.sqrt(Ham_dis)
    else:
        return 0

def main():
    Runs = pd.read_csv('Ham_dis_run_add_data.csv')
    f, ax = plt.subplots()
    df = Runs[Runs['in AF?']]
    Per = []
    Fib_beat = []
    for i in list(np.unique(df['Periodicity'])):
        Per.append(i)
        this_per = df[df['Periodicity'] == i]
        this_per['Ham_dis_80'] = this_per['Ham_dis_80'].astype(float)
        Fib_beat.append(np.mean(this_per['Ham_dis_80']))
    ax.plot(Per, Fib_beat, ls = ' ', marker = 'x')
    plt.savefig('test.png')

def Add_data(Runs):
    Runs['AF_time_real'] = False
    Runs['AF_beat'] = False
    Runs['10_before_fib'] = False
    Runs['3_before_fib'] = False
    Runs['Fib_beat'] = False
    Runs['5_after_fib'] = False
    for index, row in Runs.iterrows():
        Runs.loc[index,'Periodicity'] = int(str(row['normal_modes_config']).strip('[').split(',')[0])
        Runs.loc[index,'amp'] = float(str(row['normal_modes_config']).replace(' ', '').split(',')[1])
        Runs.loc[index,'off'] = 0.8
        if row['in AF?']:
            Runs.loc[index,'AF_time_real'] = int(str(row['AF_time']).split(',')[1].strip(')')) - 100
            Runs.loc[index,'AF_beat'] = np.floor(Runs.loc[index,'AF_time_real'] / 200)
            Runs.loc[index,'Fib_beat'] = float(str(row['Hamming_dis_arr']).split(',')[10].strip(' '))
            arr = str(row['Hamming_dis_arr']).replace(' ', '').replace('[', '').replace(']', '').split(',')
            arr_new = [float(i) for i in arr]
            Runs.loc[index,'10_before_fib'] = np.mean(arr_new[:11])
            Runs.loc[index,'3_before_fib'] = np.mean(arr_new[8:11])
            Runs.loc[index,'5_after_fib'] = np.mean(arr_new[11:])
    tot = 100*100
    Runs['Ham_dis_80'] = False
    for index, row in Runs[210:220].iterrows():
        if row['in AF?']:
            if row['AF_beat'] > 0:
                runtime = int(row['AF_time_real'])
                print(index, row['AF_beat'], runtime)
            else:
                runtime = 0
        else:
            runtime = 10000
        print(index, runtime)
        if runtime > 199:
            F = np.load('100,100,10000,0.25,25,Normal Modes,10,' + str(row['seed']) + '.npy')
            times = [80 + j*200 for j in range(int(np.floor(runtime/200) - 1))]
            print(len(times))
            Ham_dis_80 = []
            for j in times:
                time_data = F[j*tot:(j+1)*tot]
                Ham_dis_80.append(hamming_distance(time_data))
            print(Ham_dis_80)
            Runs.loc[index, 'Ham_dis_80'] = np.mean(Ham_dis_80)
        else:
            Runs.loc[index, 'Ham_dis_80'] = row['Fib_beat']
    Runs.to_csv('Ham_dis_run_add_data.csv')

def InitialDF_Ham():
    columns = list(config.keys())
    columns.extend(['seed','location_2', 'location_3', 'location_4', 'location_err', 'AF_time', 'Hamming_dis_arr', 'Ham_dis_AF', 'Ham_dis_meanx', 'per_%', 'title', 'mean', 'variance', 'in AF?', 'multiplier']) #Other columns - need animate? and fname
    columns.extend(['Ham_dis_fib_beat?'])
    columns.extend(['Max_ham_dis'])
    df = pd.DataFrame(columns=columns)
    return df
    
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

def Hamming_distance(time_data):
    activated_sites = np.where(time_data == 1)[0]
    activated_sites_x = [index_to_xy(i)[0] for i in activated_sites]
    if len(activated_sites) > 0:
        x_mean = np.mean(activated_sites_x)
        Ham_dis = np.sum((activated_sites_x-x_mean)**2)/(len(activated_sites_x))
        return np.sqrt(Ham_dis)
    else:
        return 0

def Ham_dis_inves_pdf(lattice, run):
    non_fib = []
    fib = False
    if lattice.kill:
        AF_beat = int(np.floor((lattice.AF_time[1]-100) / lattice.pacing_period))
        fib = run[19]
    else:
        AF_beat = 50
    for i in range(AF_beat):
        ham_dis_beat = []
        if lattice.AF_bool[i][1] == False:
            for j in range(200):
                time = i*200 + j
                beat_data = lattice.RefHistory[int(time*10000):int((time+1)*10000)]
                ham_dis_beat.append(Hamming_distance(beat_data))
            non_fib.append(max(ham_dis_beat))
    return fib, non_fib

def Ham_dis_pdf():
    '''
    Do every beat indivdually, say if a certain Max(HAM_DIS) is reached, then all previous values are reached as well.
    Therefore, each beat only max Ham_dis and in_AF? must be recorded.
    For AF beats, need to record hamming distance at time when fibrillation induced.
    We will get some form of a cumulative PDF
    Only want to cycle through areas with (mean-amp > percolation_threshold) to avoid trouble
    '''
    offs = np.linspace(0.4,1,13)
    amps = np.linspace(0,0.5,11)
    multi = np.linspace(1,2,11)
    off_amp_pairs = [(off, amp, mult) for off in offs for amp in amps for mult in multi if (off - amp) >= 0.4]
    periodicity = [4]#0,1,2,3,4,5,10,20
    runs = 6
    fib = []
    non_fib = []
    print(periodicity)
    for off_amp in off_amp_pairs:
        off = off_amp[0]
        amp = off_amp[1]
        mult = off_amp[2]
        print(off, amp, mult)
        for a in periodicity:
            for _ in range(runs):
                lattice = InitialLattice(x = mult)
                lattice.CouplingMethod([a,amp,off])
                run = lattice.RunIt()
                #Some Hamming distance investigation:
                fib1, non_fib1 = Ham_dis_inves_pdf(lattice, run)
                fib.append(fib1)
                non_fib.extend(non_fib1)
    np.save('fib_data_multi_{}'.format(a), fib)
    np.save('non_fib_data_multi_{}'.format(a), non_fib)

def data_hist():
    '''
    Assumptions:
    - If a simulation enters fibrillation at a hamming distance of 1, it would also enter at a Ham_dis of 2
    - If a simulation hasn't entered fibrillation with a hamming distance of 1, then it has passed through every other hamming distance between 0 and 1
    '''
    colors = {0 : 'black', 1 : 'red', 2 : 'purple', 3 : 'orange', 4 : 'gray', 5 : 'green', 10 : 'blue', 20 : 'pink'}
    for i in list(colors.keys()):
        fib_data = np.load('fib_data_multi_ref5_{}.npy'.format(i))
        fib_data = fib_data[fib_data != False]
        non_fib_data = np.load('non_fib_data_multi_ref5_{}.npy'.format(i))
        #non_fib_data = non_fib_data[abs(non_fib_data - np.mean(non_fib_data)) < 15 * np.std(non_fib_data)]
        bins = np.linspace(min(non_fib_data), max([max(fib_data)]), 100)
        prob = []
        for j in bins:
            if len(np.where(j<non_fib_data)[0]) > 0:
                prob.append(len(np.where(j>fib_data)[0])/(len(np.where(j<non_fib_data)[0]) + len(np.where(j>fib_data)[0])))
            else:
                if len(np.where(j<fib_data)[0]) > 0:
                    prob.append(1)
                else:
                    prob.append(1)
        plt.plot(bins,prob, marker = 'x', ls = ' ', color = colors[i], label = 'A = {}'.format(i), markersize = 3)
    plt.ylabel('"Fibrillation proability"')
    plt.xlabel('Hamming Distance')
    plt.legend()
    plt.savefig('Ham_dis_prob_all_ref5.png')

def data_hist1():
    fib_data = np.load('fib_data_{}.npy'.format(0))
    fib_data = fib_data[fib_data != False]
    plt.hist(fib_data)
    plt.savefig('FUNNNN.png')

def data_hist2():
    '''
    Assumptions:
    - If a simulation enters fibrillation at a hamming distance of 1, it would also enter at a Ham_dis of 2
    - If a simulation hasn't entered fibrillation with a hamming distance of 1, then it has passed through every other hamming distance between 0 and 1
    '''
    colors = {0 : 'black', 1 : 'red', 2 : 'purple', 3 : 'orange', 4 : 'gray', 5 : 'green', 10 : 'blue', 20 : 'pink'}#, 2 : 'purple', 4 : 'gray'
    for i in list(colors.keys()):
        fib_data = np.load('fib_data_multi_ref5_{}.npy'.format(i))
        fib_data = fib_data[fib_data != False]
        non_fib_data = np.load('non_fib_data_multi_ref5_{}.npy'.format(i))
        #non_fib_data = non_fib_data[abs(non_fib_data - np.mean(non_fib_data)) < 15 * np.std(non_fib_data)]
        bins = np.linspace(min(non_fib_data), max([max(fib_data)]), 50)
        diff = bins[1] - bins[0]
        prob = []
        for j in bins:
            if len(np.where(j<non_fib_data)[0]) > 0:
                prob.append(len(np.where((j<fib_data) & (j> fib_data - diff))[0])/(len(np.where(j<non_fib_data)[0]) + len(np.where((j<fib_data) & (j> fib_data - diff))[0])))
            else:
                if len(np.where(j<fib_data)[0]) > 0:
                    prob.append(1)
                else:
                    prob.append(1)
        plt.plot(bins,prob, marker = 'x', ls = ' ', color = colors[i], label = 'A = {}'.format(i), markersize = 3)
    plt.ylabel('"Fibrillation proability"')
    plt.xlabel('Hamming Distance')
    plt.legend()
    plt.savefig('Ham_dis_prob_all_ref5.png')

def chekcing_fib():
    fname = 'fib_data_multi_1.npy'
    data = np.load(fname)
    offs = np.linspace(0.4,1,13)
    amps = np.linspace(0,0.5,11)
    multi = np.linspace(1,2,11)
    off_amp_pairs = [(off, amp, mult) for off in offs for amp in amps for mult in multi if (off - amp) >= 0.4]

    dict_ = {i : [] for i in off_amp_pairs}
    k = 0
    for j in dict_.items():
        for z in range(6):
            if data[k] > 4:
                dict_[j[0]].append(data[k])
            k += 1
        if len(dict_[j[0]]) > 0:
            print(j[0])
            print(dict_[j[0]])
    

if __name__ == '__main__':
    t0 = time.time()
    data_hist2()
    #main()
    print(time.time() - t0)
