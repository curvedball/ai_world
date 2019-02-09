

'''
http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
corrected.gz 1.4MB-->47.3MB

https://www.cnblogs.com/lhuser/p/9073012.html

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


labels = np.array([4, 3, 0, 2, 1])
#labels = np.array([[0,0,0,0,1], [0,0,0,1,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]])

train_indices = np.array(list(set(np.arange(len(labels)))))


#print(train_indices)




def mini_batch_generator(sessions, labels, indices, batch_size):
    #print("Func_Hello0")
    Xbatch = np.ones((batch_size, PACKET_NUM_PER_SESSION, PACKET_LEN), dtype=np.int64) * -1
    Ybatch = np.ones((batch_size,5), dtype=np.int64) * -1
    batch_idx = 0
    #print("Func_Hello")

    while True:
        for idx in indices:
            #print("idx:%d, %d" % (idx, idx))     #zb: the first output format.
            #print("idx:{},{}".format(idx, idx))  #zb: the second output format.

            for i, packet in enumerate(sessions[idx]):
                if i < PACKET_NUM_PER_SESSION:
                    for j, byte in enumerate(packet[:PACKET_LEN]):
                        #Xbatch[batch_idx, i, (PACKET_LEN - 1 - j)] = byte
                        Xbatch[batch_idx, i, j] = byte
            #print(labels[idx])
            Ybatch[batch_idx] = np_utils.to_categorical(labels[idx], num_classes=5)  #zb: There is no [0] here!!!
            #print(Ybatch[[batch_idx]])
            batch_idx += 1

            #print("hello")
            if batch_idx == batch_size:
                batch_idx = 0
                #print(Xbatch)
                yield (Xbatch, Ybatch)
                #return


print("Func_Hello_World")
train_data_generator = mini_batch_generator(sessions, labels, train_indices, MINI_BATCH)
print(next(train_data_generator))  #zb: used with yield statment!
#print(next(train_data_generator))


#pkt0 = np.array([10, 25, 14, 3, 2, 9, 16, 76, 98, 255])