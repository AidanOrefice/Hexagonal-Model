from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Runs = pd.read_csv('Runs.csv')

freq_table = {}
freq_table_true = {}

for index, row in Runs.iterrows():
    if row['normal_modes_config'] in list(freq_table.keys()):
        freq_table[row['normal_modes_config']] += 1
        if row['normal_modes_config'] in list(freq_table_true.keys()):
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] += 1
        else:
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] = 1
    else:
        freq_table[row['normal_modes_config']] = 1
        if row['normal_modes_config'] in list(freq_table_true.keys()):
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] += 1
        else:
            if row['in AF?']:
                freq_table_true[row['normal_modes_config']] = 1

print(freq_table)
print(freq_table_true)
x = [-0.1, -0.13, -0.16, -0.19, -0.22, -0.25]
y = [float(i)/1000 for i in list(freq_table_true.values())]
f, ax = plt.subplots()
plt.bar(x,y, width = 0.01)
plt.ylabel('% of simulations that entered AF')
plt.xlabel('Alpha, (bigger values = bigger ranges)')
plt.savefig('bar' + '.png')
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