''' A Bi-directional Recurrent Neural Network implementation example
using Tensorflow library'''
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

LEARNING_RATE = 0.001
TRAINING_STEPS = 10000
BATCH_SIZE = 128

NUM_INPUT = 28
TIMESTEPS = 28
NUM_HIDDEN = 128
NUM_CLASSES = 10

X = tf.placeholder("float", [None, TIMESTEPS, NUM_INPUT])
Y = tf.placeholder("float", [None, NUM_CLASSES])

weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*NUM_HIDDEN, NUM_CLASSES]))
}
biases = {
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
}


def BiRNN(x, weights, biases):
    x = tf.unstack(x, TIMESTEPS, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias = 1.)
    lstm_bw_cell = rnn.BasicLSTMCell(NUM_CLASSES, forget_bias = 1.)
    try:
        output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell,
                            lstm_bw_cell, x, dtype = tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell,
                            x, dtype = tf.float32)
    return tf.matmul(outputs[-1], weights['out'])+weights['bias']


logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y
))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(correct_pred, tf.float32)
init = tf.global_variables_initializer()


# Convolution Neural Network 
# 利用tensorflow中的estimator
DROPOUT = 0.25


def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse = reuse):
        x = tf.reshape(x, shape = [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
        out = tf.layers.dense(fc1, n_classes)
    return out


def model_fn(features, labels, mode):
    logits_train = conv_net(features, NUM_CLASSES, DROPOUT, 
    reuse=False, is_training=True)
    logits_test = conv_net(features, NUM_CLASSES, DROPOUT, 
    reuse = False, is_training=False)
    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=Y, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})
    return estim_specs

STEPS = 100
# Build the Estimator
model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X, y=Y,
    batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=STEPS)

# DCGAN
NUM_STEPS = 2000
BATCH_SIZE = 32

IMAGE_DIM = 784
GEN_HIDDEN_DIM = 256
DISC_HIDDEN_DIM = 256
NOISE_DIM = 200

def generator(x, reuse = False):
    with tf.variable_scope('Generator', reuse = reuse):
        x = tf.layers.dense(x, units = 6*6*128)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        x = tf.layers.conv2d_transpose(x, 64, 4, 
        strides=2)
        x = tf.layers.conv2d_transpose(x, 1, 2, 
        strides=2)
        x = tf.nn.sigmoid(x)
        return x
def discriminator(x, reuse = False):
    with tf.variable_scope('Discriminator', reuse = reuse):
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, 2)
    return x

noise_input = tf.placeholder(tf.float32, shape = [None, NOISE_DIM])
real_image_input = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])

gen_sample = generator(noise_input)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse = True)
disc_concat = tf.concat([disc_real, disc_fake], axis = 0)
stacked_gan = discriminator(gen_sample, reuse = True)

disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

gen_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
z = np.random.rand(200)
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1, NUM_STEPS+1):
        batch_disc_y = np.concatenate(
            [np.ones([BATCH_SIZE]), np.zeros([BATCH_SIZE])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([BATCH_SIZE])
        feed_dict = {real_image_input:X, noise_input: z,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)

# Dynamic Recurrent Neural Network
# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
seq_max_len = 20  # Sequence max length
n_hidden = 64  # hidden layer num of features
n_classes = 2  # linear sequence or not

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):
    x = tf.unstack(x, seq_max_len, 1)
    lstm_cell = tf.contrib.rr.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, 
    x, dtype = tf.float32, sequence_length = seqlen)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    # 得到序列的最后一个输出
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddew=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
