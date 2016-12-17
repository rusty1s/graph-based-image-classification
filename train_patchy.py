import tensorflow as tf
# import numpy as np
# import os

# from model import convolutional_1d
from cifar10 import Cifar10
from data import DataSet

# Load the test data.
# PATH_DATA = './datasets/cifar10/slic_zero/2016-12-12T20-22-40'
# PATH_LABELS = './datasets/cifar10'

cifar10 = Cifar10('./datasets/cifar10')
cifar10.get_train_batch(0)

data = DataSet([
    cifar10.get_train_batch(0),
    cifar10.get_train_batch(1),
    cifar10.get_train_batch(2),
    cifar10.get_train_batch(3),
    cifar10.get_train_batch(4),
])

images, labels = data.next_batch(2)

print(images)
print(labels)
print(images.shape)
print(labels.shape)


# def get_data_and_labels(batch):
#     with open(os.path.join(PATH_DATA,
#               'data_batch_{}'.format(batch+1)), 'rb') as input:
#         data = pickle.load(input)

#     with open(os.path.join(PATH_LABELS,
#               'data_batch_{}'.format(batch+1)), 'rb') as input:
#         labels = pickle.load(input, encoding='latin1')['labels']

#     return data, labels


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

# y_conv, x, y, keep_prob = convolutional_1d(STRUCTURE)

# # **Train the Model**
# #
# # To train and evaluate the model we will use the more sphisticated ADAM
# # optimizer instead of the gradient descent optimizer.
# softmax = tf.nn.softmax_cross_entropy_with_logits(y_conv, y)
# cross_entropy = tf.reduce_mean(softmax)

# LEARNING_RATE = 1e-4
# train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# # **Evaluate the Model**
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init = tf.initialize_all_variables()

# with tf.Session() as sess:
#     sess.run(init)

#     print('Training', 60000//50, 'batches')

#     for z in range(5):
#         data, labels = get_data_and_labels(z)

#         # for i in range(10000//50):
#         for i in range(1):
#             data_batch = data[i*50:(i+1)*50]
#             labels_batch = labels[i*50:(i+1)*50]

#             new_labels = np.zeros((50, 10))
#             for j in range(len(labels_batch)):
#                 new_labels[j][labels_batch[j]] = 1.0
#             labels_batch = new_labels

#             train_step.run(feed_dict={x: data_batch, y: labels_batch,
#                                       keep_prob: 0.5})

#             if i % 5 == 0:
#                 train_accuracy = accuracy.eval(feed_dict={x: data_batch,
#                                                           y: labels_batch,
#                                                           keep_prob: 1.0})
#                 print('Step', z*200+i, 'Training accuracy', train_accuracy)
