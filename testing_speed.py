from time import time
import numpy as np

data = np.random.randint(2,size = 10000)
start = time()
for i in range(1):
    np.where(data == 1)[0]
end = time()
print(start-end)

start = time()
for i in range(1):
    np.argwhere(data == 1)
end = time()
print(start-end)
string = 'hi_{}'.format('sjkhdfgjkshldgfjkhsd')
print(string)