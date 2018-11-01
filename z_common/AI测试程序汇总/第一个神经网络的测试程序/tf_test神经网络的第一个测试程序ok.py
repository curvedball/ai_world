#coding: utf-8


'''
zb: 第一个神经网络测试程序
	参考《TensorFlow技术解析与实战》第８章
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



#定义需要处理的数据
#my_data = np.linspace(-1,1,300)
#print my_data

x_data = np.linspace(-1,1,300)[:, np.newaxis]
#print x_data

noise = np.random.normal(0, 0.05, x_data.shape)
#print noise


y_data = np.square(x_data) - 0.5 + noise   #y = x^2 – 0.5 + 噪声
#print y_data


#定义神经网络的输入变量
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])



def add_layer(inputs, in_size, out_size, activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	return outputs


#定义神经网络的层
h1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(h1, 10, 1, activation_function=None)


#定义损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))





#使用梯度下降法来训练，使得损失函数达到目标值
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)	#这里的0.1表示学习率，可以设置为更小的固定值，也可以在训练过程中来动态调整





init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data, y_data)
	plt.ion()

	for i in range(1000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 50 == 0:
			print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
			pass

	print "===================================="
	#注意: prediction与loss是关联的,loss与训练是关联的，最终训练出的模型其实就是prediction, 根据这个模型，给定x_data输入，可以产生prediction_value这样的预测值。
	prediction_value = sess.run(prediction, feed_dict={xs: x_data})  
	print prediction_value

	lines = ax.plot(x_data, prediction_value, 'r-', lw=5) 	#放大后观察也是曲线
	#ax.scatter(x_data, prediction_value, c='r', s=2)		#可以放大这个图形来观察可以看到点,　参数c表示颜色，s表示点的尺寸大小
	#plt.pause(10)
	plt.pause(0)		#图形永久停留

















