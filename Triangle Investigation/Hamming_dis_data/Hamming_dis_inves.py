import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    t0 = time.time()
    Add_data(pd.read_csv('Ham_dis_run.csv'))
    #main()
    print(time.time() - t0)
