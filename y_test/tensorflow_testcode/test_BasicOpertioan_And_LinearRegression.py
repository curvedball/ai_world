


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











































































