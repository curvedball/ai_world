

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






