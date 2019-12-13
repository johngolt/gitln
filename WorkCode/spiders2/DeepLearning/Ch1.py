import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 准备数据
trainX = np.linspace(-1, 1, 100)
trainY = 2* trainX + np.random.randn(*trainX.shape) * 0.3
# 搭建模型
X = tf.placeholde('float')
Y = tf.placeholder('float')
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')
z = tf.multiply(X, W) + b
# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
learningRate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# 迭代训练模型
init = tf.global_variables_initializer()
train_epochs = 20
display_step = 2
with tf.Session() as session:
    session.run(init)
    for epoch in range(train_epochs):
        for (x, y) in zip(trainX, trainY):
            session.run(optimizer, feed_dict={X: x, Y: y})
    # 使用模型
    session.run(z, feed_dict={X: 0.2})

'''
# 通过字典定义
inputDict = {'x': tf.placeholder('float'),
             'y': tf.placeholder('float')}

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.save(sess, 'save_path/save_file') # filename.cpkt
    saver.restore(sess, 'save_path/save_file')
'''

# 显示模型内容
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
saveDir = 'log/'
print_tensors_in_checkpoint_file(saveDir+'save_file', None, True)

# 为模型添加保存检查点
saver = tf.train.Saver(max_to_keep = 1)
with tf.Session() as sess:
    sess.run(optimizer, feed_dict={X: x, Y: y})
    saver.save(sess, saveDir + 'linearModel.cpkt', global_step = 1)
load_epoch = 18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2, saveDir + 'linearModel.cpkt' + str(load_epoch))

# 快速载入模型
cpkt = tf.train.get_checkpoint_state(saveDir)
if cpkt and cpkt.model_checkpoint_path:
    saver.restore(sess, cpkt.model_checkpoint_path)
kpt = tf.train.latest_checkpoint(saveDir)
if kpt != None:
    saver.restore(sess, kpt)


# 简便地保存检查点
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)
with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',
                                       save_checkpoint_secs = 2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():
        i = sess.run(step)
        print(i)
# 共享变量
import tensorflow as tf
with tf.variable_scope('test1'):
    var1 = tf.get_variable('firstVar', shape = [2], dtype = tf.float32)
    with tf.variable_scope('test2'):
        var2 = tf.get_variable('firstVar', shape=[2], dtype = tf.float32)
with tf.variable_scope('test1', reuse = True):
    var3 = tf.get_variable('firstVar', shape=[2], dtype = tf.float32)
    with tf.variable_scope('test2'):
        var4 = tf.get_variable('firstVar', shape=[2], dtype= tf.float32)
print(var1.name, var2.name, var3.name, var4.name)
# 作用域与操作符的受限范围
with tf.variable_scope('scope1') as sp:
    var5 = tf.get_variable('v', [1])
with tf.variable_scope('scope2'):
    with tf.variable_scope(sp) as sp1:
        var6 = tf.get_variable('v2', [1])
with tf.variable_scope('scope'):
    with tf.name_scope('bar'):
        v = tf.get_variable('v', [1])
        x = 1 + v
print('v:', v.name)
print('x.op:', x.op.name)
