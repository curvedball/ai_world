# -*- coding: utf-8 -*-

'''
from numpy import *
=======
import numpy as np
import pandas as pd



print('Hello world2!')


# 两行三列
v2 = array([[1, 1, 2], [1, 1, 0]])
print(v2)




from numpy import *
print('hello')





import tensorflow as tf
print(tf.__version__)


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



#a = np.random.rand(3, 4)
#print(a)

a = np.random.randn(3, 5)
print(a)



from numpy import *
print('hello')

import tensorflow as tf
print(tf.__version__)



#===================================================================
#https://blog.csdn.net/hhhuua/article/details/79989822

import input_data
from tensorflow.examples.tutorials.mnist import input_data
print('hello')


import tensorflow as tf
print(tf.__version__)
print(tf.__path__)




import tensorflow as tf
#a=tf.constant(10)
#a = tf.zeros(shape=[1,2])
#a
x=tf.Variable(tf.ones([3,3]))
y=tf.Variable(tf.zeros([3,3]))
#init=tf.initialize_all_variables()
init=tf.global_variables_initializer
x = tf.placeholder(tf.float32, [None, 784])



import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # type
input2 = tf.placeholder(tf.float32)  # type

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.0]}))








import tensorflow as tf
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
'''


'''
import tensorflow as tf
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run([product])
print(result)





import tensorflow as tf
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    #with tf.device("/gpu:1"):
    #with tf.device("/cpu:0"):
    with tf.device("/cpu:10000"):
        result = sess.run([product])
print(result)





import tensorflow as tf
x = tf.Variable(3)
y = tf.Variable(5)
z=x+y
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(z))




import tensorflow as tf
word=tf.constant('hello,world!')
with tf.Session() as sess:
    print(sess.run(word))




import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print('a+b=',sess.run(add, feed_dict={a: 2, b: 3}))
    print('a*b=',sess.run(mul, feed_dict={a: 2, b: 3}))






import tensorflow as tf
a=tf.Variable(tf.ones([3,2]))
b=tf.Variable(tf.ones([2,3]))
#product=tf.matmul(5*a,4*b)
product=tf.matmul(tf.multiply(5.0,a), tf.multiply(4.0,b))
#init=tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(product))
'''


'''
import tensorflow as tf

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.
one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph. We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# Launch the graph and run the ops.
with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)
    # Print the initial value of 'state'
    #print(sess.run(state))


    # Run the op that updates 'state' and print 'state'.
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# output:
# 0
# 1
# 2
# 3
'''




'''
import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
print(result)

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]




import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)


    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))


# 输出:
# [array([ 14.], dtype=float32)]



import tensorflow as tf
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
with tf.Session() as sess:
    print(sess.run(adder_node, {a: 3, b:4.5}))
    print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))




import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model, {x:[1,2,3,4]}))




#===================================Loss Function==========================

#https://blog.csdn.net/xiaopihaierletian/article/details/61923808


import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(linear_model, {x:[1,2,3,4]}))
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))






import tensorflow as tf

W = tf.Variable([-1.], tf.float32)
b = tf.Variable([1.], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(linear_model, {x:[1,2,3,4]}))
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
'''





'''
import tensorflow as tf

#W = tf.Variable([.3, .3, .3, .3], tf.float32)
#b = tf.Variable([-.3, -.3, -.3, -.3], tf.float32)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init) # reset values to incorrect defaults.
    #for i in range(100):
    for i in range(500):
    #for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
        print(sess.run([W, b]))
'''




'''
#===================================Loss Function: Linear Regression==========================
#import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
sess.close()
'''



'''
#===================================Loss Function: Linear Regression Call API==========================
import tensorflow as tf

# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)


# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)


# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)


# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))
'''



'''
#===================================Loss Function: Linear Regression===DIY API==========================
import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x'] + b

    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)


estimator = tf.contrib.learn.Estimator(model_fn=model)

# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)


estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn, steps=10))
'''








'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#if *.gz is not exist, then download it, otherwise load it directly!
mnist = input_data.read_data_sets("/nn/a/", one_hot = True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
'''





'''
#==========================Simple Neural Network (Hand written Digit Recognition)==========================
import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data

#zb: 注意，这里使用了one-hot编码，即如果手写的数字的类别是３，则转换为0001000000，one-hot编码编码通过一个标志位来表示数字
mnist = input_data.read_data_sets("/nn/a/", one_hot=True)

#放置占位符，用于在计算时接收输入值,每个为784维向量，下面的定义表示二维浮点数的一个张量，形状[无，784]这里没有一个维度可以任意长。
x = tf.placeholder(tf.float32, [None, 784])

#为了进行训练，需要把正确值一并传入网络
#zb: 注意y_actual对一个样本的类别值已经转换为一行二进制的值，比如类别是３，则对应0001000000
y_actual = tf.placeholder(tf.float32, shape=[None, 10])


#创建两个变量，分别用来存放权重值W和偏置值b，对于机器学习，一般模型的参数都是变量。
W = tf.Variable(tf.zeros([784, 10]))        #初始化权值W,这里的初始值为0
b = tf.Variable(tf.zeros([10]))            #初始化偏置项b 初始值为0
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)     #加权变换并进行softmax回归，得到预测概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法以0.01的学习率最小化交叉熵使得残差最小

#zb: correct_prediction矩阵成为m*1的矩阵，含有若干个１
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值


#初始化之前创建的变量的操作
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

#启动初始化
with tf.Session() as sess:
    sess.run(init)

    #开始训练模型，循环1000次，每次都会随机抓取训练数据中的100条数据，然后作为参数替换之前的占位符来运行train_step
    for i in range(1000):               #训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)           #按批次训练，每批100行数据
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})   #执行训练

        if (i % 100 == 0):                  #每训练100次，测试一次, zb: 准确率只有86.73%
            print("accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))

'''








#===========================================CNN (To understand)==========================================
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf

# 导入input_data用于自动下载和安装MNIST数据集
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("/nn/a/", one_hot=True)  # 下载并加载mnist数据

# 创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder(tf.float32, [None, 784])  # 输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 输入的标签占位符


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    # 输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层

# x 是一个4维张量，shape为[batch,height,width,channels]

# 卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层

# 采用最大池化，也就是取窗口中的最大值作为结果

# x 是一个4维张量，shape为[batch,height,width,channels]#ksize表示pool窗口大小为2x2,也就是高2，宽2

# strides，表示在height和width维度上的步长都为2
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建网络

# 把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]

# -1表示自动推测这个维度的size,表示一次输入计算的数量
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中

# 初始化W为[5,5,1,32]的张量，表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
W_conv1 = weight_variable([5, 5, 1, 32])

# 初始化b为[32],即输出大小
b_conv1 = bias_variable([32])

# 把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling

# h_pool1的输出即为第一层网络输出，shape为[batch,14,14,1]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
h_pool1 = max_pool(h_conv1)  # 第一个池化层

# 第2层，卷积层

# 卷积核大小依然是5*5，这层的输入和输出神经元个数为32和64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# h_pool2即为第二层网络输出，shape为[batch,7,7,1]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool2 = max_pool(h_conv2)  # 第二个池化层

# 第3层, 全连接层

# 这层是拥有1024个神经元的全连接层

# W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# 计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量-1代表的含义是不用我们自己指定这一维的大小，函数会自动算，
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层

# Dropout层

# 为了减少过拟合，在输出层前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

# 输出层

# 最后，添加一个softmax层

# 可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层

# 预测值和真实值之间的交叉墒
cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 交叉熵

# train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法

# 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。

# 因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))

# 计算正确预测项的比例，因为tf.equal返回的是布尔值，

# 使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算

# 创建一个交互式Session
sess = tf.InteractiveSession()

# 初始化变量
sess.run(tf.initialize_all_variables())

# 开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:  # 训练100次，验证一次
        # 每100次输出一次日志

        train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
        print('step', i, 'training accuracy', train_acc)
        train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("test accuracy", test_acc)




















































