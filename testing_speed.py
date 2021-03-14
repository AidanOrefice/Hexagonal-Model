from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Animation import Animate
from Hexagon import HexagonalLattice
from matplotlib.lines import Line2D
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

mpl.rcParams.update({
    'figure.figsize' : [16,9],
    'xtick.labelsize' : 15,
    'ytick.labelsize' : 15,
    'axes.labelsize' : 25,
    'legend.fontsize' : 17,
    'savefig.bbox' : 'tight',
})

def plot_amp_offs_PS(fnames):
    f, ax = plt.subplots(3,2, sharey = True, sharex = True, figsize=(16,13.5))
    print(ax)
    print(ax[0])
    for t,fname in enumerate(fnames):
        ax_num = (int(np.floor(t/2)),int(t%2))
        A = str(fname.split('_')[-1].split('.')[0])
        print(A)
        if A != '756':
            Runs = pd.read_csv(fname)
            for index, row in Runs.iterrows():
                list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
                Runs.loc[index, 'normal_beta'] = float(list_[2].strip())
                Runs.loc[index, 'normal_alpha'] = float(list_[1].strip())
            
            Runs = Runs[Runs['normal_alpha'] < 0.51]

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
            if int(A) != 10:
                im = ax[ax_num[0],ax_num[1]].imshow(df)
            else:
                im = ax[ax_num[0],ax_num[1]].imshow(df)
            ax[ax_num[0],ax_num[1]].tick_params(axis='y', rotation=0)
            CS = ax[ax_num[0],ax_num[1]].contour([i for i in range(len(map_beta.keys()))],[i for i in range(len(map_alpha.keys()))], Per_percent, levels = [0,0.5,0.99], colors = 'blue', alpha = 1)
            ax[ax_num[0],ax_num[1]].clabel(CS, inline=1, fontsize=7.5)
            ax[ax_num[0],ax_num[1]].set_title('A = {}'.format(A))
    for ax1 in ax.flat:
        ax1.set_yticks([0,2,4,6,8,10])
        ax1.set_yticklabels([0.5,0.4,0.3,0.2,0.1,0])
        ax1.set_xticks([2*i for i in range(11)])
        ax1.set_xticklabels([i/10 for i in range(11)])
    cbar_ax = f.add_axes([0.93, 0.15, 0.03, 0.7])
    f.colorbar(im, cax=cbar_ax)
    f.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Offset')
    plt.ylabel('Amplitude')
    #ax[2].set_xticklabels(ax[2].get_xticks(), rotation = 90)
    plt.savefig('PS_poster_1_5')

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
        for j in range(100-len(times)):
            times.append(10000)
        if len(times) > 0:
            beat = np.mean(times)//200
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

def ham_dis_fun(fnames):
    Runs = {}
    offs = [0.4,0.5,0.6,0.7,0.8]
    for i in fnames:
        Run = pd.read_csv(i)
        Run = Run[Run['location_err']]
        Run = Run[Run['in AF?']]
        Run['AF_time_real'] = False
        Run['AF_beat'] = False
        Run['10_before_fib'] = False
        Run['3_before_fib'] = False
        Run['Fib_beat'] = False
        Run['5_after_fib'] = False
        for index, row in Run.iterrows():
            Run.loc[index,'Periodicity'] = int(str(row['normal_modes_config']).strip('[').split(',')[0])
            Run.loc[index,'amp'] = float(str(row['normal_modes_config']).replace(' ', '').split(',')[1])
            Run.loc[index,'off'] = float(str(row['normal_modes_config']).replace(' ', '').replace(']', '').split(',')[2])
            if row['in AF?']:
                Run.loc[index,'AF_time_real'] = int(str(row['AF_time']).split(',')[1].strip(')')) - 100
                Run.loc[index,'AF_beat'] = np.floor(Run.loc[index,'AF_time_real'] / 200)
                Run.loc[index,'Fib_beat'] = float(str(row['Hamming_dis_arr']).split(',')[10].strip(' '))
                arr = str(row['Hamming_dis_arr']).replace(' ', '').replace('[', '').replace(']', '').split(',')
                arr_new = [float(i) for i in arr]
                Run.loc[index,'10_before_fib'] = np.mean(arr_new[:11])
                Run.loc[index,'3_before_fib'] = np.mean(arr_new[8:11])
                Run.loc[index,'5_after_fib'] = np.mean(arr_new[11:])
        Runs[float(i.split('_')[1])] = Run
    
    Mean_Ham_dis = []
    for i in offs:
        Mean_Ham_dis.append(np.mean(Runs[i]['Fib_beat']))
    
    f, ax = plt.subplots()
    ax.plot(offs, Mean_Ham_dis, ls = '', marker = 'x')
    ax.set_xlabel('Offset')
    ax.set_ylabel('Average Hamming distance before fibrillation')
    plt.savefig('Ham_dis_test_.png')

def Ham_Dis_plot(fname):
    Runs = pd.read_csv(fname)
    Runs_no_fib = Runs[Runs['Ham_dis_fib_beat?'] == False]
    Ham_dis_avg = [np.mean(Runs_no_fib[Runs_no_fib.loc[:,str(i)] != 'False'].loc[:,str(i)].astype(float)) for i in range(200)]
    time = [i for i in range(200)]
    f, ax = plt.subplots()
    ax.plot(time, Ham_dis_avg, color = 'black', label = 'Non-fibrillatory beats')
    Runs_fib = Runs[Runs['Ham_dis_fib_beat?'] == True]
    for i in [0,1,2,3,4]:
        Ham_dis_avg_fib = []
        time1 = []
        Runs_no_fib_i = Runs_fib.iloc[i]
        for j in range(200):
            x = str(Runs_no_fib_i.loc[str(j)])
            if x != 'False':
                Ham_dis_avg_fib.append(float(x))
                time1.append(j)
        ax.plot(time1, Ham_dis_avg_fib, color = 'red', label = 'Fibrillatory beats_{}'.format(i))
    plt.legend()
    plt.savefig('Ham_dis_test__.png')

def Ham_Dis_plot_per_dep(fname):
    Runs = pd.read_csv(fname)
    Runs = Runs[Runs['Ham_dis_fib_beat?'] == False]
    periodicity = [1,3,5,10,20]
    time = [i for i in range(200)]
    f, ax = plt.subplots()
    Runs['periodicity'] = Runs['normal_modes_config'].str.extract('(\d+)')
    colors = {0 : 'black', 1: 'red', 3 : 'blue', 5 : 'green', 10 : 'orange', 20 : 'purple'}
    for i in periodicity:
        Runs_p = Runs[Runs.periodicity == str(i)]
        Ham_dis_avg = [np.mean(Runs_p[Runs_p.loc[:,str(j)] != 'False'].loc[:,str(j)].astype(float)) for j in range(200)]
        ax.plot(np.asarray(time[0:int(100/i)])*i, Ham_dis_avg[0:int(100/i)], color = colors[i], label = 'periodicity {}'.format(i))
    ax.set_xlabel('Time since last activation')
    ax.set_ylabel('Average Hamming distance')
    plt.legend()
    plt.savefig('Ham_dis_test__.png')

def Ham_Dis_plot_per1(fname):
    Runs = pd.read_csv(fname)
    periodicity = [i for i in range(1, 21)]
    f, ax = plt.subplots()
    Runs['periodicity'] = Runs['normal_modes_config'].str.extract('(\d+)')
    print(Runs.periodicity)
    colors = {0 : 'black', 1: 'red', 3 : 'blue', 5 : 'green', 10 : 'orange', 20 : 'purple'}
    Ham_dis_avg = {i : [] for i in list(colors.keys())}
    xmean = {i : [] for i in list(colors.keys())}
    for index, row in Runs.iterrows():
        periodicity = float(row.periodicity)
        if periodicity in list(colors.keys()):
            Ham_dis_avg[periodicity].append(row['Ham_dis_AF'])
            xmean[periodicity].append(row['Ham_dis_meanx'])
    for i in list(colors.keys()):
        ax.plot(xmean[i], Ham_dis_avg[i], color = colors[i], ls = '', marker = 'x', label = 'Periodicty of {}'.format(i))
    plt.legend()
    plt.savefig('Ham_dis_test_periodicity.png')

def Merge_data(fnames):
    Runs = [0 for i in fnames]
    for i,j in enumerate(fnames):
        Runs[i] = pd.read_csv(j)
    Run_tot = pd.concat(Runs)
    for index, row in Run_tot.iterrows():
        Run_tot.loc[index,'Periodicity'] = int(str(row['normal_modes_config']).strip('[').split(',')[0])
        Run_tot.loc[index,'amp'] = float(str(row['normal_modes_config']).replace(' ', '').split(',')[1])
        Run_tot.loc[index,'off'] = float(str(row['normal_modes_config']).replace(' ', '').replace(']', '').split(',')[2])
    Run_tot.to_csv('Ham_dis_run_All_Data.csv')

def Ham_Dis_plot_per(fname):
    Runs = pd.read_csv(fname)
    Runs = Runs[Runs['Ham_dis_fib_beat?'] == False]
    Runs = Runs[Runs['location_err']]
    periodicity = 1
    off = np.unique(Runs['off'])
    amp = 0.2
    time = [i for i in range(200)]
    f, ax = plt.subplots()
    colors = {0.4 : 'black', 0.45: 'red', 0.5 : 'grey', 0.55 : 'blue', 0.6: 'green', 0.65 : 'yellow', 0.7 : 'orange', 0.75 : 'black', 0.8 : 'blue', 0.85 : 'cyan', 0.8999999999999999  : 'brown', 0.95 : 'purple', 1: 'pink'}
    for i in off:
        Runs_p = Runs[Runs.Periodicity == periodicity]
        Runs_p = Runs_p[Runs_p.amp == amp]
        Runs_p = Runs_p[Runs_p.off == i]
        if len(Runs_p) > 0:
            Ham_dis_avg = [np.mean(Runs_p[Runs_p.loc[:,str(j)] != 'False'].loc[:,str(j)].astype(float)) for j in range(200)]
            ax.plot(time, Ham_dis_avg, color = colors[i], label = 'Offset {}'.format(i))
    ax.set_xlabel('Time since last activation')
    ax.set_ylabel('Average Hamming distance')
    plt.legend()
    plt.savefig('Ham_dis_test__.png')

def SigmoidDist(charges):
        return 1/(1+np.exp(-25 *(charges-0.25)))

def sigmoid_plot():
    values = [i/6 for i in range(7)]
    x = np.linspace(0,1,51)
    y = [SigmoidDist(i) for i in x]
    bars = [SigmoidDist(i) for i in values]
    print(bars)
    print(values)
    plt.bar(values, bars, color = 'white', width = 0.00001, edgecolor = 'black')
    values1 = [r'$\frac{%i}{6}$'%(i) for i in range(7)]
    values1 = ['{}/6'.format(i) for i in range(7)]
    plt.xticks(values, values1)
    plt.plot(x,y, color = 'red')
    plt.xlim(-0.05,1.05)
    plt.ylim(0,1.19)
    plt.ylabel('Probability of activation')
    plt.xlabel('Charge')
    legend_elements = [Line2D([0], [0], ls='-', color='red', label='Charge activation function',  markersize=1),
                Line2D([0], [0], ls='-', color='black', label='Discrete charge values', markersize=1)]
    plt.legend(handles = legend_elements)
    plt.savefig('Poster_sigmoid')

#Ham_Dis_plot_per('Ham_dis_run_All_Data.csv')
#Merge_data(['Ham_dis_run_fib_PS_{}.csv'.format(i) for i in [1,2,3,4,5,6]])
#ham_dis_fun(['FailureMultiplierData_0.4_full.csv', 'FailureMultiplierData_0.5_full.csv', 'FailureMultiplierData_0.6_full.csv','FailureMultiplierData_0.7_full.csv','FailureMultiplierData_0.8_full.csv'])

#SigmoidPlot('FailureMultiplierData_1.csv')



'''for i in ['FailureMultiplierData_0.8_20.csv']:
    SigmoidPlot(i)'''

#sigmoid_plot()
fname = ['Normal_Modes_Phase_Space_Ham_dis_1.csv','Normal_Modes_Phase_Space_Ham_dis_5.csv','Normal_Modes_Phase_Space_Ham_dis_2.csv','Normal_Modes_Phase_Space_Ham_dis_10.csv','Normal_Modes_Phase_Space_Ham_dis_3.csv','Normal_Modes_Phase_Space_Ham_dis_20.csv']
plot_amp_offs_PS(fname)

'''
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