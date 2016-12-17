import os
import tensorflow as tf
# import numpy as np
# import os

from model import convolutional_1d
from data import DataSet

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle

# Load the test data.
PATH_DATA = './datasets/cifar10/slic_zero/2016-12-17T15-22-11'
PATH_LABELS = './datasets/cifar10'


with open(os.path.join(PATH_DATA, 'data_batch_1'), 'rb') as f:
    batch_1 = pickle.load(f)
with open(os.path.join(PATH_DATA, 'data_batch_2'), 'rb') as f:
    batch_2 = pickle.load(f)
with open(os.path.join(PATH_DATA, 'data_batch_3'), 'rb') as f:
    batch_3 = pickle.load(f)
with open(os.path.join(PATH_DATA, 'data_batch_4'), 'rb') as f:
    batch_4 = pickle.load(f)
with open(os.path.join(PATH_DATA, 'data_batch_5'), 'rb') as f:
    batch_5 = pickle.load(f)

data = DataSet([
    batch_1,
    batch_2,
    batch_3,
    batch_4,
    batch_5,
])

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
        train_step.run(feed_dict={x: d, y: labels, keep_prob: 0.5})

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: d,
                                                      y: labels,
                                                      keep_prob: 1.0})
            print('Step', i, 'Training accuracy', train_accuracy)
