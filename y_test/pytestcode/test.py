
import numpy as np
import pandas as pd

'''
print('Hello world2!')

a = np.random.randn(5, 1)
print (a)
print(a.T)
print(a.shape)

b = a.T
print(b)
'''




'''
#About 3ms
print('Hello world3!')

import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

#print('time consumed: %s ms.\n' % ((toc - tic)*1000))
print('time consumed: ' + str((toc - tic)*1000) + ' ms.')
'''


'''
#about 300ms
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)
c = 0

tic = time.time()
for i in range(1000000):
    c = c + a[i] * b[i]
toc = time.time()
print('time consumed: ' + str((toc - tic)*1000) + ' ms.')
'''


#a = np.random.rand(3, 4)
#print(a)

a = np.random.randn(3, 5)
print(a)

