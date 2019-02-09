

'''
http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
corrected.gz 1.4MB-->47.3MB

'''
import numpy as np
from keras.utils import np_utils

MINI_BATCH = 2
PACKET_NUM_PER_SESSION = 3
PACKET_LEN = 10

pkt0 = np.array([10, 25, 14, 3, 2, 9, 16, 76, 98, 255])
pkt1 = np.array([11, 25, 14, 3, 2, 9, 16, 76, 98, 255])
pkt2 = np.array([12, 25, 14, 3, 2, 9, 16, 76, 98, 255])
pkt3 = np.array([13, 25, 14, 3, 2, 9, 16, 76, 98, 255])
pkt4 = np.array([14, 25, 14, 3, 2, 9, 16, 76, 98, 255])


sess0 = []
sess0.append(pkt0)
sess0.append(pkt1)
sess0.append(pkt2)


sess1 = []
sess1.append(pkt1)
sess1.append(pkt2)
sess1.append(pkt3)


sess2 = []
sess2.append(pkt2)
sess2.append(pkt3)
sess2.append(pkt4)



sess3 = []
sess3.append(pkt3)
sess3.append(pkt4)
sess3.append(pkt0)


sess4 = []
sess4.append(pkt4)
sess4.append(pkt0)
sess4.append(pkt1)




sessions = []
sessions.append(sess0)
sessions.append(sess1)
sessions.append(sess2)
sessions.append(sess3)
sessions.append(sess4)
#print(sessions)

#print(sess0)
#print(sessions[0])
#print(sessions[1])


#labels = np.array([4, 3, 0, 2, 4])
labels = np.array([[0,0,0,0,1], [0,0,0,1,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]])


print(np_utils.to_categorical(labels[0], num_classes=5))


arr = [10,20,30,40,50,60]
for index,value in enumerate(arr):
    print ('%s,%s' % (index, value))