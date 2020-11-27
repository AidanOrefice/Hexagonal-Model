from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Runs = pd.read_csv('Trial_Varying_Variance_PS_0.25.csv')

for index, row in Runs.iterrows():
    list_ = str(row['normal_modes_config']).split('[')[1].split(']')[0].split(',')
    Runs.loc[index, 'normal_beta'] = float(list_[3].strip())
    Runs.loc[index, 'normal_alpha'] = float(list_[2].strip())

map_alpha = {j:i for i,j in enumerate(np.unique(Runs['normal_alpha']))}
map_beta = {j:i for i,j in enumerate(np.unique(Runs['normal_beta']))}

print(map_alpha)
print(map_beta)

tot_count = Runs['normal_alpha'].value_counts()[0]/len(map_beta.keys())
print(tot_count)
PS = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
PS_time = np.zeros((len(map_alpha.keys()),len(map_beta.keys())))
print(PS.shape)
for index, row in Runs.iterrows():
    if row['in AF?']:
        x = map_beta[row['normal_beta']]
        y = map_alpha[row['normal_alpha']]
        PS[y][x] += 1
        PS_time[y][x] += row['%time in AF']
PS_per = PS / tot_count
PS_time = PS_time / tot_count
df = pd.DataFrame(PS_per, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
df.index = np.round(df.index*100)/100
df.columns = np.round(df.columns*100)/100
f, ax = plt.subplots()
sns.heatmap(df, ax = ax)
ax.tick_params(axis='y', rotation=0)
ax.set_xlabel('Mean')
ax.set_ylabel('Amplitude')
ax.set_title('% time simulation entered AF')
plt.savefig('PS_25_colourmap')

df = pd.DataFrame(PS_time, columns = list(map_beta.keys()), index = list(map_alpha.keys()))
df.index = np.round(df.index*100)/100
df.columns = np.round(df.columns*100)/100
f, ax = plt.subplots()
sns.heatmap(df, ax = ax)
ax.tick_params(axis='y', rotation=0)
ax.set_xlabel('Mean')
ax.set_ylabel('Amplitude')
ax.set_title('Average % time simulations in spent AF')
plt.savefig('PS_25_colourmap_time')

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
'''data = np.random.randint(2,size = 10000)
start = time()
for i in range(100000):
    np.where(data == 1)[0]
end = time()
print(start-end)

start = time()
for i in range(100000):
    data == 1
end = time()
print(start-end)
string = 'hi_{}'.format('sjkhdfgjkshldgfjkhsd')
print(string)'''