

'''
http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
corrected.gz 1.4MB-->47.3MB

'''
import numpy as np



'''
from struct import *
file = open(r"/zb/cnn/test.bin", "wb")
file.write(pack("idh", 12345, 67.89, 15))
file.close()
'''

'''
from struct import *
file = open(r"/zb/cnn/test.bin", "rb")
(a,b,c) = unpack("idh",file.read(8+8+2))
print(a)
print(b)
print(c)
'''


'''
file = open(r"/zb/cnn/test.bin", "rb")
s=file.read(1)
byte=ord(s)
print(hex(byte))

s=file.read(1)
byte=ord(s)
print(hex(byte))
'''



from struct import *
s = open(r"/zb/cnn/test.bin", "rb").read(1)
#print(unpack('c', s))
print(unpack('b', s))






