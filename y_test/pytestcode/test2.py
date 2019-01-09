

#https://www.jianshu.com/p/4ed7f7b15736





#===============================Part 1================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf





#===============================Part 2================================================================
# 定义全局变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'





#===============================Part 3================================================================
# 检测本地是否有数据集
def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir # /tmp/cifar10_data
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # 从URL中获得文件名
    filename = DATA_URL.split('/')[-1]

    # 合并文件路径
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 定义下载过程中打印日志的回调函数
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # 下载数据集
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print()

        # 获得文件信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        # 解压缩
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#
maybe_download_and_extract()





#===============================Part 4================================================================
# 删除之前训练过程中产生的一些临时文件，并重新生成目录
if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
gfile.MakeDirs(FLAGS.train_dir)


#定义记录训练步数的变量
global_step = tf.train.get_or_create_global_step()   # tf.Variable(0, trainable=False)






#===============================Part 5================================================================

# 从 CIFAR-10 中导入数据和标签
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def distorted_inputs(batch_size):
    """
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # 如果没有设置data_dir，抛出异常
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    # 合并解压后的数据路径
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    # 要读入的数据文件
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    # 如果有数据文件缺失，抛出异常
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 把要读取的全部文件打包为一个tf内部的queue类型，之后tf开文件就从这个queue中取目录
    filename_queue = tf.train.string_input_producer(filenames)

    # 读取文件队列中文件的样本
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 用于训练网络的图像处理，请注意应用于图像的许多随机失真
    # 随机裁剪图像的[height, width]部分
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # 随机水平翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 由于这些操作是不可交换的，因此可以考虑随机化和调整操作的顺序
    # 在某范围随机调整图片亮度
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    # 在某范围随机调整图片对比度
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # 减去平均值并除以像素的方差，白化操作：均值变为0，方差变为1
    float_image = tf.image.per_image_standardization(distorted_image)

    # 设置张量的形状.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # 确保随机shuffling有好的混合性质
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # 通过建立一个样本队列来生成一批image和label
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)


def read_cifar10(filename_queue):
    """
    读取和解析来自CIFAR10数据文件的样本
    建议：如果您想要N路并行读取，请调用此函数N次
    这会给你N个独立的Readers，阅读那些文件中不同的文件和位置，这将提供更好的混合例子

    ARGS：
    filename_queue：具有要读取的文件名的字符串队列。

    返回：
    表示单个样本的对象，包含以下字段：
    height：结果中的行数（32）
    width：结果中的列数（32）
    depth：结果中的颜色通道数量（3）
    key：描述这个例子文件名和记录号的标量字符串张量
    label：一个int32张量，带有范围为0..9的标签
    uint8image：一个图像数据的[height, width, depth] uint8 张量
    """

    # 定义返回的结果对象类
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    # 只有10个类别
    label_bytes = 1  # 2 for CIFAR-100
    # 32x32 RGB 的图像
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    # 每条记录的格式固定：label+image，因此长度固定
    record_bytes = label_bytes + image_bytes

    # 采用固定长度的阅读器，CIFAR-10格式没有文件头或文件尾，将header_bytes和footer_bytes保留为默认值0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    # 打开filename_queue中的文件，读取一条记录
    result.key, value = reader.read(filename_queue)
    # 阅读器的read方法会输出一个key来表征输入的文件和其中的纪录(对于调试非常有用)
    # 同时得到一个字符串标量，这个字符串标量可以被一个或多个解析器，或者转换操作将其解码为张量并且构造成为样本。

    # 将字符串标量转换为长度为record_bytes的uint8张量
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 第一个字节代表了label，类型转换uint8->int32，与一般的切片操作不同，tf.slice的第三个参数是切片的长度
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 标签之后的剩余字节表示图像，reshape [depth * height * width] => [depth，height，width]
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # 交换输入张量的不同维度 [depth, height, width] => [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    """
    构建排队的一批图像和标签
    ARGS：
     image：type.float32的[height，width，3]的3-D张量
     label：type.int32的1-D张量
     min_queue_examples：int32，在队列中保留的最小样本数量，可提供多批样本
     batch_size：每批次的图像数量

     返回：
     images: Images. 4D张量 [batch_size，height，width，3]
     labels: Labels. 1D张量 [batch_size]
    """
    # 创建一个混合样本的队列，然后从样本队列中读取batch_size的图像+标签
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                 num_threads=num_preprocess_threads,
                                                 capacity=min_queue_examples + 3 * batch_size,
                                                 min_after_dequeue=min_queue_examples)
    # 在数据输入管线的末端,我们需要有另一个队列来执行输入样本的训练(train)，评价(loss)和推理(inference)
    # 因此我们使用tf.train.shuffle_batch函数来对队列中的样本进行乱序处理

    # 在可视化器中显示训练图像
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


images, labels = distorted_inputs(batch_size=FLAGS.batch_size)





#==============================Part 6=================================================================

# 尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求
TOWER_NAME = 'tower'


def inference(images):
    """
    构建CIFAR-10模型
    ARGS：
     images：从distorted_inputs（）或inputs（）返回的图像
    返回：
     Logits
    """
    # 我们使用tf.get_variable（）而不是tf.Variable（）来实例化所有变量，以便跨多个GPU训练时能共享变量
    # 如果我们只在单个GPU上运行此模型，我们可以通过用tf.Variable（）替换tf.get_variable（）的所有实例来简化此功能

    # conv1-第一层卷积
    with tf.variable_scope('conv1') as scope:  # 每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀
        # 5*5 的卷积核，64个
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)
        # 卷积操作，步长为1，0padding SAME，不改变宽高，通道数变为64
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        # 在CPU上创建第一层卷积操作的偏置变量
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        # 加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # relu非线性激活
        conv1 = tf.nn.relu(bias, name=scope.name)
        # 创建激活显示图的summary
        _activation_summary(conv1)

    # pool1-第一层pooling
    # 3*3 最大池化，步长为2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1-局部响应归一化
    # LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2-第二层卷积
    with tf.variable_scope('conv2') as scope:
        # 卷积核：5*5 ,64个
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # norm2-局部响应归一化
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2-第二层最大池化
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3-全连接层，384个节点
    with tf.variable_scope('local3') as scope:
        # 把单个样本的特征拼成一个大的列向量，以便我们可以执行单个矩阵乘法
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

        # 权重
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        # 偏置
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        # relu激活
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # 生成summary
        _activation_summary(local3)

    # local4-全连接层，192个节点
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # softmax, i.e. softmax(WX + b)
    # 输出层
    with tf.variable_scope('softmax_linear') as scope:
        # 权重
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        # 偏置
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        # 输出层的线性操作
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        # 生成summary
        _activation_summary(softmax_linear)
    return softmax_linear


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    帮助创建一个权重衰减的初始化变量

    请注意，变量是用截断的正态分布初始化的
    只有在指定了权重衰减时才会添加权重衰减

    Args:
    name: 变量的名称
    shape: 整数列表
    stddev: 截断高斯的标准差
    wd: 加L2Loss权重衰减乘以这个浮点数.如果没有，此变量不会添加权重衰减.

    Returns:
    变量张量
    '''
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    '''
    帮助创建存储在CPU内存上的变量
    ARGS：
     name：变量的名称
     shape：整数列表
     initializer：变量的初始化操作
    返回：
     变量张量
    '''
    with tf.device('/cpu:0'):  # 用 with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _activation_summary(x):
    '''
    为激活创建summary

    添加一个激活直方图的summary
    添加一个测量激活稀疏度的summary

    ARGS：
     x：张量
    返回：
     没有
    '''
    # 如果这是多GPU训练，请从名称中删除'tower_ [0-9] /'.这有助于张量板上显示的清晰度.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


logits = inference(images)









#=============================Part 7 ==================================================================
# 描述损失函数，往inference图中添加生成损失（loss）所需要的操作（ops）
def loss(logits, labels):
    '''
    将L2Loss添加到所有可训练变量
    添加"Loss" and "Loss/avg"的summary
    ARGS：
    logits：来自inference（）的Logits
    labels：来自distorted_inputs或输入（）的标签.一维张量形状[batch_size]

    返回：
    float类型的损失张量
    '''

    labels = tf.cast(labels, tf.int64)
    # 计算这个batch的平均交叉熵损失
    # 添加一个tf.nn.softmax_cross_entropy_with_logits操作，用来比较inference()函数所输出的logits Tensor与labels
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # 总损失定义为交叉熵损失加上所有的权重衰减项（L2损失）
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, labels)







#===============================Part 8================================================================


# 描述模型的训练
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.


def train(total_loss, global_step):
    '''
    训练 CIFAR-10模型

    创建一个optimizer并应用于所有可训练变量. 为所有可训练变量添加移动平均值.
    ARGS：
     total_loss：loss()的全部损失
     global_step：记录训练步数的整数变量
    返回：
     train_op：训练的op
    '''
    # 影响学习率的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 根据步骤数以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # Summary是对网络中Tensor取值进行监测的一种Operation.这些操作在图中是“外围”操作，不影响数据流本身.
    # 把lr添加到观测中
    tf.summary.scalar('learning_rate', lr)

    # 生成所有损失和相关和的移动平均值的summary
    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # 应用梯度.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 为可训练变量添加直方图summary.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 为梯度添加直方图summary
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # 跟踪所有可训练变量的移动平均值
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def _add_loss_summaries(total_loss):
    '''
    往CIFAR-10模型中添加损失summary
    为所有损失和相关summary生成移动平均值，以便可视化网络的性能

    ARGS：
     total_loss：loss()的全部损失
    返回：
     loss_averages_op：用于生成移动平均的损失
    '''
    # 计算所有单个损失和总损失的移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # 把所有的单个损失和总损失添加到summary观测中，平均损失也添加观测
    for l in losses + [total_loss]:
        # 将每个损失命名为损失的原始名称+“（raw）”，并将损失的移动平均版本命名为损失的原始名称
        # 这一行代码应该已经过时了，执行时提醒：
        # INFO:tensorflow:Summary name conv1/weight_loss (raw) is illegal; using conv1/weight_loss__raw_ instead.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


train_op = train(loss, global_step)





#==================================Part 9=============================================================

# 开始训练

# 用tf.train.Saver()创建一个Saver来管理模型中的所有变量
saver = tf.train.Saver(tf.all_variables())

# 获取所有监测的操作'sum_opts',基于TF collection中的Summaries构建summary操作
summary_op = tf.summary.merge_all()

# 初始化所有的变量.
init = tf.initialize_all_variables()

# 运行计算图中的所有操作.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
sess.run(init)

# 调用run或者eval去执行read之前，必须调用tf.train.start_queue_runners来将文件名填充到队列.否则read操作会被阻塞到文件名队列中有值为止
tf.train.start_queue_runners(sess=sess)

# 指定监测结果输出目录
summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=sess.graph_def)


for step in xrange(FLAGS.max_steps):
    # 记录运行计算图一次的时间
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 100 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

    if step % 1000 == 0:
        # 添加summary日志
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

    # 定期保存模型检查点
    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

























































