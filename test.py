import numpy as np


k = [i*(np.pi)/4 for i in range(17) ]
A = 10
print(A/pow(2,0.5))
for i in k:
    x = np.linspace(0,i,100)
    print('ends at %f pi ' %(i/np.pi))
    y = A*np.sin(x)
    print(np.std(y))

