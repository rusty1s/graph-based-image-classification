# TODO das dataset als parameter h

import tensorflow as tf

from data import (Cifar10, inputs)
from .helper import (weight_variable, bias_variable)
from .logger_hook import LoggerHook

cifar10 = Cifar10()


def train_op(total_loss, global_step):
    lr = learning_rate(50000, 128, global_step, 350.0, 0.1, 0.1)
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss)
    return opt


def learning_rate(num_examples_per_epoch, batch_size, global_step,
                  num_epochs_per_decay, initial_learning_rate,
                  learning_rate_decay_factor):

    num_batches_per_epoch = num_examples_per_epoch/batch_size
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

    # When training a model, it is often recommended to lower the learning rate
    # as the training progresses. This function applies an exponential decay
    # function to a provided initial learning rate.
    # We apply a discrete decay of the learning rate by passing staircase as
    # True.
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               learning_rate_decay_factor,
                                               staircase=True)

    # tf.contrib.deprecated.scalar_summary('learning_rate', learning_rate)

    return learning_rate


def inference(images):
    """Build the model"""

    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable([5, 5, 3, 64])
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = bias_variable([64])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope('local2') as scope:
        reshape = tf.reshape(norm1, [128, -1])
        dim = reshape.get_shape()[1].value
        weights = weight_variable([dim, 384])
        biases = bias_variable([384])
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable([384, 10])
        biases = bias_variable([10])

        softmax_linear = tf.add(tf.matmul(local2, weights), biases,
                                name=scope.name)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = inputs(cifar10, batch_size=128)

        logits = inference(images)
        ls = loss(logits, labels)

        op = train_op(ls, global_step)

        with tf.train.MonitoredTrainingSession(
                hooks=[
                    tf.train.StopAtStepHook(last_step=1000),
                    tf.train.NanTensorHook(ls),
                    LoggerHook(ls, batch_size=128, display_step=10)]
                ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(op)

        print(labels)
        print(images)
