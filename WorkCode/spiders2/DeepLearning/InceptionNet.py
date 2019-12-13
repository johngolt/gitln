import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0., stddev=stddev)
def inception_v3_arg_scope(weight_decay = 0.00004,
                           stddev = 0.1,
                           batch_norm_var_collection = 'moving_var'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
        weights_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weight_initializer = tf.truncated_normal_initializer(stddev = stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params
        ) as sc:
            return sc

def inception_v3_base(input, scope = None):
    end_points = { }
    with tf.variable_scope(scope, 'Inceptionv3', [input]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d,
                             slim.arg_pool2d], stride = 1, padding = 'VALID'):
            net = slim.conv2d(input, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding = 'SAME',
                              scope = 'Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope = 'Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3, 3], scope = 'Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3, 3], stride = 2,
                                  scope = 'MaxPool_5a_3x3')

        with slim.arg_scope([slim.conv2d, slim.max_pool2d,
                             slim.avg_pool2d], stride = 1, padding = 'SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5],
                                           scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3],
                                               scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5],
                                           scope = 'Conv1_0c_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3],
                                               scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2,
                                 branch_3], 3)

            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5],
                                           scope = 'Conv1_0c_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3],
                                               scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2,
                                 branch_3], 3)

            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2, padding = 'VALID',
                                           scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [5, 5],
                                           scope = 'Conv1_0c_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2,
                                           padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1],
                                           scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3],
                                               scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1],
                                           scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2,
                                 branch_3], 3)
