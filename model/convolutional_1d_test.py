from nose.tools import *

import tensorflow as tf
import numpy as np

from .convolutional_1d import (output_width, convolutional_1d)


def test_output_width():
    pass
    # The convolution and max pooling operations have padding set to 'SAME', so
    # that we have always no-zero padded outputs.

    # We have an input width of 7 and a stride size of 2. That means after
    # convolution our input width is reduced to 4. A pooling of stride 1
    # shouldn't change this.
    width = output_width(7, 2, 1)
    assert_equals(width, 4)

    # We have an input width of 10 and a stride size of 1. That means after 
    # convolution our input width is reduced to 10. A pooling of stride 2
    # should reduce this to 5.
    width = output_width(10, 1, 2)
    assert_equals(width, 5)

    # We have an input width of 10 and a stride size of 4. That means after 
    # convolution our input width is reduced to 3. A pooling of stride 2 should 
    # reduce this to 2.
    width = output_width(10, 4, 2)
    assert_equals(width, 2)

    # We have an input width of 10 and a stride size of 2. That means after
    # convolution our input width is reduced to 5. A pooling of stride 1
    # shouldn't change this.
    width = output_width(10, 2, 1)
    assert_equals(width, 5)


def test_convolutional_1d_a():
    structure = {
        'in_width': 10,
        'in_channels': 1,
        'conv': [
            {'patch': 2, 'stride': 2, 'out_channels': 32, 'max_pool': 2},
        ],
        'full': [
            64,
        ],
        'out': 2,
    }

    y_conv, x, y, keep_prob = convolutional_1d(structure)

    softmax = tf.nn.softmax_cross_entropy_with_logits(y_conv, y)
    cross_entropy = tf.reduce_mean(softmax)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        data = np.empty((1, 1, 10, 1))
        labels = np.zeros((1, 2))
        labels[0][1] = 1.0

        train_step.run(feed_dict={x: data, y: labels, keep_prob: 0.5})
        accuracy.eval(feed_dict={x: data, y: labels, keep_prob: 1.0})


def test_convolutional_1d_b():
    structure = {
        'in_width': 100,
        'in_channels': 5,
        'conv': [
            {'patch': 10, 'stride': 5, 'out_channels': 32, 'max_pool': 2},
            {'patch': 4, 'stride': 2, 'out_channels': 64, 'max_pool': 2},
        ],
        'full': [
            124,
            1024,
        ],
        'out': 10,
    }

    y_conv, x, y, keep_prob = convolutional_1d(structure)

    softmax = tf.nn.softmax_cross_entropy_with_logits(y_conv, y)
    cross_entropy = tf.reduce_mean(softmax)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        data = np.empty((2, 1, 100, 5))
        labels = np.zeros((2, 10))
        labels[0][4] = 1.0
        labels[1][7] = 1.0

        train_step.run(feed_dict={x: data, y: labels, keep_prob: 0.5})
        accuracy.eval(feed_dict={x: data, y: labels, keep_prob: 1.0})
