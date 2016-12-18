import os
import tensorflow as tf
# import numpy as np
# import os

from model import convolutional_1d
from data import load_data

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle

# Load the train and test data.
PATH = './datasets/cifar10/slic_zero/2016-12-17T15-22-11'
train_set = load_data(PATH, ['data_batch_{}'.format(i) for i in range(1, 6)])
test_set = load_data(PATH, ['test_batch'])

# Build the model.
NEIGHBORHOOD_SIZE = 9
NUM_FEATURES = 6
NODE_SIZE = 100

STRUCTURE = {
    'in_width': NODE_SIZE * NEIGHBORHOOD_SIZE,
    'in_channels': NUM_FEATURES,
    'conv': [
        {
            'patch': NEIGHBORHOOD_SIZE,
            'stride': NEIGHBORHOOD_SIZE,
            'out_channels': 32,
            'max_pool': 2,
        },
        {
            'patch': 3,
            'stride': 1,
            'out_channels': 64,
            'max_pool': 2,
        },
    ],
    'full': [
        1024,
    ],
    'out': 10,
}

y_conv, x, y, keep_prob = convolutional_1d(STRUCTURE)

# **Train the Model**
#
# To train and evaluate the model we will use the more sphisticated ADAM
# optimizer instead of the gradient descent optimizer.
softmax = tf.nn.softmax_cross_entropy_with_logits(y_conv, y)
cross_entropy = tf.reduce_mean(softmax)

LEARNING_RATE = 1e-4
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# **Evaluate the Model**
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

print('START TRAINING')

with tf.Session() as sess:
    sess.run(init)

    for i in range(20000):
        d, labels = data.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: d,
                                                      y: labels,
                                                      keep_prob: 1.0})
            print('Step', i, 'Training accuracy', train_accuracy)

        train_step.run(feed_dict={x: d, y: labels, keep_prob: 0.5})
