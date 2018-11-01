#coding: utf-8


'''
zb: 第一个神经网络测试程序
	参考《TensorFlow技术解析与实战》第８章
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



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
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data, y_data)
	plt.ion()

	for i in range(1000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  #喂入的数据就是训练数据
		if i % 50 == 0:
			print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
			pass
	print "===================================="
	prediction_value = sess.run(prediction, feed_dict={xs: x_data})
	print prediction_value

	lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
	#plt.pause(10)
	plt.pause(0)

















