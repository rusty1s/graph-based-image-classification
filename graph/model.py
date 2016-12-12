import tensorflow as tf
import numpy as np

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle


# **Weight Initialization**
#
# To create the model, we're going to need to create a lot of weights and
# biases. One should generally initialize weights with a small amount of noise
# for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU
# neurons, it is also good practice to initialize them with a slightly positive
# initial bias to avoid "dead neurons". Instead of doing this repeatedly while
# we build the model, let's create two handy functions to do it for us.
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# **Convolution and Pooling**
#
# Given an input tensor of shape [batch, in_width, in_channels] and a
# filter/kernel tensor of shape [filter_width, in_channels, out_channels], this
# operation reshapes the arguments to pass them to conv2d to perform the
# equivalent convolution operation.
#
# * input: A 4D Tensor. Must be of type float32 or float64.
# * filters: A 4D Tensor. Must have the same type as input.
# * stide: An integer. The number of entries by which the filter is moved right
#   at each step.
def conv1d(input, filters, stride):
    return tf.nn.conv2d(input, filters, strides=[1, 1, stride, 1],
                        padding='SAME')


# Our pooling is plain old max pooling over 1x2 blocks for a 1D convolution.
#
# * input: A 4D Tensor with shape [batch, height, width, channels] and type
#   tf.float32.
# * ksize: A list of ints that has length >= 4. The size of the window for each
#   dimension of the input tensor.
# * strides: A list of ints that has length >= 4. The stride of the sliding
#   window for each dimension of the input tensor.
def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                          padding='SAME')


# A input of a 1D convolution has the shape [in_width, in_channels]. So the
# input we get from the graph should be reshaped to a 1D vector for each
# feature. The stride size is the size of the neighborhood.
#
# From the graph we get the shape [in_channels, node_size, neighborhood_size].
# We need to convert it to [node_size * neighborhood_size, in_channels].
def convert_input(input):
    num_features = len(input)
    num_nodes = len(input[0])
    num_neighbors = len(input[0][0])

    # Reshape to [in_channels, node_size * neighborhood_size]
    input = input.reshape((num_features, num_nodes * num_neighbors))

    # Swap the axes.
    input = np.swapaxes(input, 0, 1)

    new_array = np.zeros((1, 1, 900, 6))
    new_array[0][0] = input

    return new_array

NEIGHBORHOOD_SIZE = 9
NUM_FEATURES = 6
NODE_SIZE = 100

# Graph variables
x = tf.placeholder(tf.float32,
                   [None, 1, NODE_SIZE*NEIGHBORHOOD_SIZE, NUM_FEATURES])
y = tf.placeholder(tf.float32, [None, 10])

# ** First Convolutional Layer**
#
# We an now implement our first layer. It will consist of convolution, followed
# by max pooling. The convolution will compute 32 features for each
# neighborhood size. Its weight tensor will have a shape of [neighborhood_size,
# in_channels, 32]. The first dimension is the patch size, the next is the
# number of input channels, and the last is the number of output channels. We
# will also have a bias vector with a component for each output channel.
W_conv1 = weight_variable([1, NEIGHBORHOOD_SIZE, NUM_FEATURES, 32])
b_conv1 = bias_variable([32])

# We then convolve the input with the weight tensor, add the bias, apply the
# ReLU function, and finally max pool. The max_pool_1x2 method will reduce the
# input to NODE_SIZE/2.
h_conv1 = tf.nn.relu(conv1d(x, W_conv1, NEIGHBORHOOD_SIZE) + b_conv1)
h_pool1 = max_pool(h_conv1)

# ** Second Convolutional Layer**
#
# In order to build a deep network, we stack several layers of this type. The
# second layer will have 64 features for each patch. For the second patch phase
# we choose a 1x5 patch size with a stride of 1.
SECOND_PATCH_SIZE = 5
W_conv2 = weight_variable([1, SECOND_PATCH_SIZE, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2, 1) + b_conv2)
h_pool2 = max_pool(h_conv2)

# After the first convolutional layer we had a size of NODE_SIZE/2 for each
# graph input and for every of the 32 features. After applying convolution on a
# 1x5 patch with stride size 1 the graph has been reduced to NODE_SIZE/2 - 4.
# THIRD_INPUT_SIZE = int(NODE_SIZE/2) - (SECOND_PATCH_SIZE - 1)
# TODO: WHY NODE_SIZE/4????
THIRD_INPUT_SIZE = int(NODE_SIZE/4)

# **Fully Connected Layer**
#
# We add a fully-connected layer with 1024 neurons to allow processing on the
# entire graph.
W_fc1 = weight_variable([THIRD_INPUT_SIZE * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool1, [-1, THIRD_INPUT_SIZE * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# **Dropout**
#
# To reduce overfitting, we will apply dropout before the readout layer. We
# create a placeholder for the probability that a neuron's output is kept
# during dropout. This allows us to turn dropout on during training, and turn
# if off during testing.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# **Readout Layer**
#
# Finally, we add our output layer of ten classes
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
#
# To train and evaluate the model we will use the more sphisticated ADAM
# optimizer instead of the gradient descent optimizer.

# Define cost function and training step.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,
                                                                       y))
LEARNING_RATE = 1e-4
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

# Load the test data
PATH_DATA = './datasets/cifar10/slic_zero/2016-12-12T20-22-40/data_batch_1'
PATH_LABELS = './datasets/cifar10/data_batch_1'

with open(PATH_DATA, 'rb') as input:
    datas = pickle.load(input)

with open(PATH_LABELS, 'rb') as input:
    labels = pickle.load(input, encoding='latin1')['labels']

with tf.Session() as sess:
    sess.run(init)

    for i in range(0, 10000):
        data = np.zeros((1, 1, 900, 6))
        data[0] = datas[i]
        print(data)
        label = np.zeros((1, 10))
        label[0][int(labels[i])] = 1

        train_step.run(feed_dict={
            x: data,
            y: label,
            keep_prob: 0.5
        })

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: data,
                y: label,
                keep_prob: 1.0,
            })

            print('Step {}, training accuracy {}'.format(i, train_accuracy))
