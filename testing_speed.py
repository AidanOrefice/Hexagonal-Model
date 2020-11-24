from time import time
import numpy as np
import pandas as pd

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