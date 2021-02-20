from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Animation import Animate
from Hexagon import HexagonalLattice
from matplotlib.lines import Line2D

def plot_amp_offs_PS(fname):
    Runs = pd.read_csv(fname)
    A = str(fname.split('_')[-1].split('.')[0])

    for index, row in Runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        Runs.loc[index, 'normal_beta'] = float(list_[2].strip())
        Runs.loc[index, 'normal_alpha'] = float(list_[1].strip())

    map_alpha = {j:i for i,j in enumerate(sorted(np.unique(Runs['normal_alpha']), reverse = True))}
    map_beta = {j:i for i,j in enumerate(np.unique(Runs['normal_beta']))}

    tot_count = Runs['normal_alpha'].value_counts()[0]/len(map_beta.keys())
    print(tot_count)
    PS = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    Per_percent = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    #PS_time = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    print(PS.shape)
    for index, row in Runs.iterrows():
        x = map_beta[row['normal_beta']]
        y = map_alpha[row['normal_alpha']]
        if row['in AF?']:
            PS[y][x] += 1
        Per_percent[y][x] += row['per_%']
        #PS_time[y][x] += row['%time in AF']
    PS_per = PS / tot_count
    Per_percent = Per_percent / tot_count
    #PS_time = PS_time / tot_count
    df = pd.DataFrame(PS_per, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
    df.index = np.round(df.index*100)/100
    df.columns = np.round(df.columns*100)/100
    f, ax = plt.subplots()
    sns.heatmap(df, ax = ax)
    ax.tick_params(axis='y', rotation=0)
    CS = ax.contour([i for i in range(len(map_beta.keys()))],[i for i in range(len(map_alpha.keys()))], Per_percent, levels = [0,0.5,0.99], colors = 'blue', alpha = 1)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel('Offset')
    ax.set_ylabel('Amplitude')
    ax.set_title('Fraction of simulations that entered fibrillation')
    plt.savefig('PS_25_large_colourmap_{}'.format(A))

def plot_per_percent_PS(fname):
    Runs = pd.read_csv(fname)
    A = str(fname.split('_')[-1].split('.')[0])

    for index, row in Runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        Runs.loc[index, 'normal_beta'] = float(list_[2].strip())
        Runs.loc[index, 'normal_alpha'] = float(list_[1].strip())

    map_alpha = {j:i for i,j in enumerate(sorted(np.unique(Runs['normal_alpha']), reverse = True))}
    map_beta = {j:i for i,j in enumerate(np.unique(Runs['normal_beta']))}

    print(map_alpha)
    print(map_beta)

    tot_count = Runs['normal_alpha'].value_counts()[0]/len(map_beta.keys())
    print(tot_count)
    PS = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    Per_S = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    #PS_time = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    print(PS.shape)
    for index, row in Runs.iterrows():
        x = map_beta[row['normal_beta']]
        y = map_alpha[row['normal_alpha']]
        Per_S[y][x] += row['per_%']
        PS[y][x] += 1
            #PS_time[y][x] += row['%time in AF']
    PS_per = Per_S / PS
    #PS_time = PS_time / tot_count
    df = pd.DataFrame(PS_per, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
    df.index = np.round(df.index*100)/100
    df.columns = np.round(df.columns*100)/100
    f, ax = plt.subplots()
    sns.heatmap(df, ax = ax)
    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Offset')
    ax.set_ylabel('Amplitude')
    ax.set_title('Fraction of wavefronts that Percolate')
    plt.savefig('PS_per_percent_{}'.format(A))

def plot_mean_var_PS(fname):
    Runs = pd.read_csv(fname)
    A = str(fname.split('_')[-1].split('.')[0])
    print(A)
    for index, row in Runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        Runs.loc[index, 'normal_beta'] = float(list_[2].strip())
        Runs.loc[index, 'normal_alpha'] = float(list_[1].strip())

    Runs = Runs[Runs['normal_alpha'] < 0.51]
    Runs['std'] = np.sqrt(Runs['variance'])

    Runs['std_cut'] = pd.cut(Runs['std'], list(np.linspace(-0.01, 0.31, 17)), labels = list(np.linspace(0, 0.3, 16)))
    Runs['mean_cut'] = pd.cut(Runs['mean'], list(np.linspace(-0.01, 1.01, 27)), labels = list(np.linspace(0, 1, 26)))

    map_alpha = {j:i for i,j in enumerate(sorted(np.unique(Runs['std_cut']), reverse = True))}
    map_beta = {j:i for i,j in enumerate(np.unique(Runs['mean_cut']))}

    print(map_alpha)
    print(map_beta)

    PS = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    PS_tot = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
    for index, row in Runs.iterrows():
        x = map_beta[row['mean_cut']]
        y = map_alpha[row['std_cut']]
        PS_tot[y][x] += 1
        if row['in AF?']:
            PS[y][x] += 1
            #PS_time[y][x] += row['%time in AF']
        
    PS_per = PS / PS_tot
    df = pd.DataFrame(PS_per, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
    df.index = np.round(df.index*100)/100
    df.columns = np.round(df.columns*100)/100
    f, ax = plt.subplots()
    sns.heatmap(df, ax = ax)
    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Std')
    ax.set_title('Fraction of simulations that entered fibrillation')
    plt.savefig('PS_mean_std_large_colourmap_{}'.format(A))

def SigmoidPlot(fname):
    Runs = pd.read_csv(fname)
    df = Runs[Runs['in AF?']]
    fun = np.unique(Runs['multiplier'])
    avg_time = []
    re_plot = []
    off = fname.split('_')[-1].split('.c')[0]
    lens = {}
    for i in fun:
        vals = df.loc[df['multiplier'] == i]['AF_time']
        times = [int(k.split(' ')[-1].split(')')[0])-100 for k in vals]
        for j in range(200-len(times)):
            times.append(10000)
        if len(times) > 0:
            beat = np.median(times)//200
            print(beat)
            avg_time.append(beat)
            re_plot.append(i)
            lens[i] = len(times)
    print(avg_time)
    print(lens)
    plt.figure(figsize=(16,9))
    plt.plot(re_plot,avg_time, ls = ' ', marker = 'x')
    plt.xlabel('Multiplier on Misfire Probability', fontsize = 18)
    plt.ylabel('Average Number of Beats before Fib', fontsize = 18)
    plt.title('')

    plt.xticks(fontsize =15)
    plt.yticks(fontsize =15)

    label_offs = 'Offset = ' + str(off)
    label_amp = 'Amplitude = 0.2'

    legend_elements = [Line2D([0], [0], marker='o', color='white', label=label_offs, markerfacecolor='white', markersize=0),
                Line2D([0], [0], marker='o', color='white', label=label_amp, markerfacecolor='white', markersize=0)]
    plt.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=3, fontsize = 14)

    plt.savefig('SigMultiPlot_{}.png'.format(off))



def plot_amp_offs_periodicity():
    Runs = pd.read_csv('PeriodicityInvestigation.csv')

    for index, row in Runs.iterrows():
        list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
        Runs.loc[index, 'A1_x'] = float(list_[0].strip())
        Runs.loc[index, 'A2_y'] = float(list_[1].strip())
        Runs.loc[index, 'normal_beta'] = float(list_[3].strip())
        Runs.loc[index, 'normal_alpha'] = float(list_[2].strip())

    map_A1_x = {j:i for i,j in enumerate(sorted(np.unique(Runs['A1_x']), reverse = True))}
    map_A2_y = {j:i for i,j in enumerate(np.unique(Runs['A2_y']))}

    print(map_A1_x)
    print(map_A2_y)

    #tot_count = Runs['A1_x'].value_counts()[0]/len(map_A2_y.keys())
    #print(tot_count)
    PS = np.zeros((len(map_A1_x.keys()),len(map_A2_y.keys())))
    #PS_time = np.zeros((len(map_A1_x.keys()),len(map_A2_y.keys())))
    print(PS.shape)
    for index, row in Runs.iterrows():
        if row['in AF?']:
            x = map_A2_y[row['A2_y']]
            y = map_A1_x[row['A1_x']]
            PS[y][x] += 1
            #PS_time[y][x] += row['%time in AF']
    PS_per = PS / 10 #Need to hard code number of runs
    #PS_time = PS_time / tot_count
    df = pd.DataFrame(PS_per, columns = list(map_A2_y.keys()), index = list(map_A1_x.keys()))
    df.index = np.round(df.index*100)/100
    df.columns = np.round(df.columns*100)/100
    f, ax = plt.subplots()
    sns.heatmap(df, ax = ax)
    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Periodicity in A2_y')
    ax.set_ylabel('Periodicity in A1_x')
    ax.set_title('Fraction of simulations that entered fibrillation')
    plt.savefig('Periodicity_heatmap.png')

#SigmoidPlot('FailureMultiplierData_1.csv')

for i in ['FailureMultiplierData_0.7.csv']:
    SigmoidPlot(i)

'''
fname = ['Normal_Modes_Phase_Space_20.csv','Normal_Modes_Phase_Space_10.csv','Normal_Modes_Phase_Space_5.csv','Normal_Modes_Phase_Space_3.csv','Normal_Modes_Phase_Space_1.csv']
for i in fname:
    plot_amp_offs_PS(i)

df = pd.DataFrame(PS_time, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
df.index = np.round(df.index*100)/100
df.columns = np.round(df.columns*100)/100
f, ax = plt.subplots()
sns.heatmap(df, ax = ax)
ax.tick_params(axis='y', rotation=0)
ax.set_xlabel('Offset')
ax.set_ylabel('Amplitude')
ax.set_title('Average % time simulations in spent AF')
plt.savefig('PS_25_large_colourmap_time')'''
'''
def run_and_animate_random(row):
    lattice = HexagonalLattice(row['width'],
        row['height'],
        row['runtime'],
        row['threshold'],
        row['sigmoid_strength'],
        row['coupling'],
        row['refractory_period'],
        True,
        'transition',
        row['stats'],
        seed = row['seed'])
    
    lattice.CreateLattice()
    input_config = row['normal_modes_config'].split('[')[1].split(']')[0].split(',')
    lattice.CouplingMethod(row['constant'], row['gradient'], row['normal_modes'], input_config,
     row['grad_start'], row['grad_end'] )
    Ref_His, bool_in, i = lattice.RunIt()
    name = (str(row['width']) + "," + str(row['height']) + "," + str(row['runtime']) + "," + str(row['threshold']) +
     "," + str(row['sigmoid_strength']) + "," + "Normal Modes" + "," + "," + str(row['refractory_period']) + "," )
    if bool_in:
        Animate(Ref_His, i, name)
print(Runs.iloc[4970])
run_and_animate_random(Runs.iloc[4970])'''

'''freq_table = {}
freq_table_true = {}
time_in_af = {}

for index, row in Runs.iterrows():
    if row['normal_modes_config'] in list(freq_table.keys()):
        freq_table[row['normal_modes_config']] += 1
        if row['normal_modes_config'] in list(freq_table_true.keys()):
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] += 1
                time_in_af[row['normal_modes_config']] += row['%time in AF']
        else:
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] = 1
                time_in_af[row['normal_modes_config']] = row['%time in AF']
    else:
        freq_table[row['normal_modes_config']] = 1
        if row['normal_modes_config'] in list(freq_table_true.keys()):
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] += 1
                time_in_af[row['normal_modes_config']] += row['%time in AF']
        else:
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] = 1
                time_in_af[row['normal_modes_config']] = row['%time in AF']

print(freq_table)
print(freq_table_true)
x = [0,0.05,0.1,0.15,0.2,0.25,0.3]
y = [float(i)/200 for i in list(freq_table_true.values())]
f, ax = plt.subplots()
plt.bar(x,y, width = 0.01)
plt.ylabel('% of simulations that entered AF')
plt.xlabel('Alpha, (bigger values = bigger ranges)')
plt.savefig('bar_5' + '.png')

y = [float(i)/200 for i in list(time_in_af.values())]
f, ax = plt.subplots()
plt.bar(x,y, width = 0.01)
plt.ylabel('% time spent in AF')
plt.xlabel('Alpha, (bigger values = bigger ranges)')
plt.savefig('bar_5_1' + '.png')'''