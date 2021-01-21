from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Animation import Animate
from Hexagon import HexagonalLattice

Runs = pd.read_csv('Prelim.csv')

for index, row in Runs.iterrows():
    list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
    Runs.loc[index, 'normal_beta'] = float(list_[3].strip())
    Runs.loc[index, 'normal_alpha'] = float(list_[2].strip())

map_alpha = {j:i for i,j in enumerate(sorted(np.unique(Runs['normal_alpha']), reverse = True))}
map_beta = {j:i for i,j in enumerate(np.unique(Runs['normal_beta']))}

print(map_alpha)
print(map_beta)

tot_count = Runs['normal_alpha'].value_counts()[0]/len(map_beta.keys())
print(tot_count)
PS = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
#PS_time = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
print(PS.shape)
for index, row in Runs.iterrows():
    if row['in AF?']:
        x = map_beta[row['normal_beta']]
        y = map_alpha[row['normal_alpha']]
        PS[y][x] += 1
        #PS_time[y][x] += row['%time in AF']
PS_per = PS / tot_count
#PS_time = PS_time / tot_count
df = pd.DataFrame(PS_per, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
df.index = np.round(df.index*100)/100
df.columns = np.round(df.columns*100)/100
f, ax = plt.subplots()
sns.heatmap(df, ax = ax)
ax.tick_params(axis='y', rotation=0)
ax.set_xlabel('Mean')
ax.set_ylabel('Amplitude')
ax.set_title('Fraction of simulations that entered fibrillation')
plt.savefig('PS_25_large_colourmap')

'''df = pd.DataFrame(PS_time, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
df.index = np.round(df.index*100)/100
df.columns = np.round(df.columns*100)/100
f, ax = plt.subplots()
sns.heatmap(df, ax = ax)
ax.tick_params(axis='y', rotation=0)
ax.set_xlabel('Mean')
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