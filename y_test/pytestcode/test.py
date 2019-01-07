
<<<<<<< HEAD
'''
from numpy import *
=======
import numpy as np
import pandas as pd
>>>>>>> 92317ccf6502a2027d5ac662a64e33a9048ae205

'''
print('Hello world2!')

<<<<<<< HEAD
# 两行三列
v2 = array([[1, 1, 2], [1, 1, 0]])
print(v2)
'''


'''
from numpy import *
print('hello')
'''


import tensorflow as tf
print(tf.__version__)

=======
a = np.random.randn(5, 1)
print (a)
print(a.T)
print(a.shape)
>>>>>>> 92317ccf6502a2027d5ac662a64e33a9048ae205

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

