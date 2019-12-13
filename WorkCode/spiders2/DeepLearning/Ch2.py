# 图的基本操作
import numpy as np
import tensorflow as tf
# 建立图
c = tf.constant(0, name='first') # 启动默认图
g = tf.Graph() # 建立新图
with g.as_default(): # 新图的作用域
    c1 = tf.constant(0, name = 'first')
    print(c1.graph)
    print(g)
    print(c.graph)
g2 = tf.get_default_graph() # 得到当前的默认图
print(g2)
tf.reset_default_graph() # 重新设置默认图, 启动一个新图，并设为默认
g3 = tf.get_default_graph()
print(g3)

# 获得张量
t = g.get_tensor_by_name(name = 'first:0')
print(t)
# 获取节点操作
a = tf.constant([1, 2])
b = tf.constant([3, 4])
tensor1 = tf.matmul(a, b, name = 'first')
test = g3.get_tensor_by_name('first:0')
print(tensor1.op.name) # 得到操作的名字
testop = g3.get_operation_by_name('first')
with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    test = tf.get_default_graph().get_tensor_by_name('first:0')
    print(test)
# 获取元素列表
tt2 = g.get_operations()
# 获取对象：根据对象来获取元素。
tt3 = g.as_graph_element(c1) # 传入一个对象， 返回一个张量或是一个op.

print(tf.get_default_graph())
g = tf.Graph()
with g.as_default():
    print(g)
    print(tf.get_default_graph())

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]
# 计算交叉熵
result = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
logits_scaled = tf.nn.softmax(logits)
result2 = -tf.reduce_mean(labels * tf.log(logits_scaled), 1)
labels2 = [2, 1] # 从0开始， 表明labels2中总共分为3个类
# 先对labels2进行one-hot encoding 然后计算交叉熵
result3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels2, logits = logits)
loss = tf.reduce_mean(result) # 计算损失值
with tf.Session() as sess:
    print(sess.run(result), sess.run(result2), sess.run(result3))

# 退化学习率
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step = global_step,
                                           decay_steps=10, decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):
        g, rate = sess.run([add_global, learning_rate])
        print(g, rate)
