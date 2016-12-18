import tensorflow as tf
import math

from .helper import (weight_variable, bias_variable)


# Given an input tensor of shape [batch, 1, in_width, in_channels] and a
# filter/kernel tensor of shape [1, filter_width, in_channels, out_channels],
# this operation performs a 2D convolution with the shape of a 1D convolution.
# The convolution is rounded up with SAME padding.
#
# * input: A 4D Tensor. Must be of type float32 or float64.
# * filters: A 4D Tensor. Must have the same type as input.
# * stide: An integer. The number of entries by which the filter is moved right
#   at each step.
def conv1d(input, filters, stride):
    return tf.nn.conv2d(input, filters, strides=[1, 1, stride, 1],
                        padding='SAME')


# Our pooling is plain old max pooling over a defined width with the same
# stride width. The max pooling is rounded up with SAME padding.
def max_pool(input, size):
    return tf.nn.max_pool(input, ksize=[1, 1, size, 1],
                          strides=[1, 1, size, 1], padding='SAME')


# Helper method that computes the output_width of a 1D convolution after max
# pooling. It gets the input width and the stride width of the convolution and
# the width of the pooling operation.
def output_width(input_width, stride, pool):
    after_conv = math.ceil(float(input_width)/stride)
    return int(math.ceil(after_conv/pool))


# Builds a 1d convolutional neural net. The `structure` object models the
# structure of the 1d convolutional neural net. It is build upon five keys:
# 'in_width', 'in_channels', conv', 'full', 'out'. 'conv' and 'full' are
# optional.
#
# Example:
#
# {
#   'in_width',
#   'in_channels',
#   'conv': [
#     {patch, stride, out_channels, max_pool},
#     {patch, stride, out_channels, max_pool},
#   ],
#   'full': [
#     out_width,
#     out_width,
#   ],
#   'out',
# }
#
# The model creates three placeholders:
#
# * x - the input: [None, 1, in_width, in_channels].
# * y - the output: [None, out_channels].
# * keep_prob: The probability that a neuron's output is kept during dropout.
#   This allows us to turn dropout on during training, and turn if off during
#   testing.
def convolutional_1d(structure):
    # Placeholder variables that gets overridden over time.
    in_channels = structure['in_channels']
    in_width = structure['in_width']

    # Create the graph placeholders.
    x = tf.placeholder(tf.float32, [None, 1, in_width, in_channels])
    y = tf.placeholder(tf.float32, [None, structure['out']])
    keep_prob = tf.placeholder(tf.float32)

    # Placeholder variable that gets overridden over time.
    input = x

    if 'conv' in structure:
        for layer in structure['conv']:
            patch = layer['patch']
            stride = layer['stride']
            out_channels = layer['out_channels']
            pool = layer['max_pool']

            # We can now incrementally implement our convolutional layers. They
            # will consist of convolution followed by max pooling. The
            # convolution will compute `layer['out_channels']` features for
            # each patch size defined by `layer['patch_size']`. We will also
            # have a bias vector with a component for each output channel.
            W = weight_variable([1, patch, in_channels, out_channels])
            b = bias_variable([out_channels])

            # We then convolve the input with the weight tensor, add the bias,
            # apply the ReLU function, and finally max pool.
            h_conv = tf.nn.relu(conv1d(input, W, stride))
            input = max_pool(h_conv, pool)

            # We save attributes for the next layer.
            in_channels = out_channels

            # Compute the input width for the next layer.
            in_width = output_width(in_width, stride, pool)

    # We add fully-connected layers to allow processing on the entire graph.
    in_width = in_width * in_channels
    input = tf.reshape(input, [-1, in_width])

    if 'full' in structure:
        for out_width in structure['full']:
            W = weight_variable([in_width, out_width])
            b = bias_variable([out_width])

            input = tf.nn.relu(tf.matmul(input, W) + b)

            # We save attributes for the next layer.
            in_width = out_width

    # To reduce overfitting, we will apply dropout before the readout layer.
    h_drop = tf.nn.dropout(input, keep_prob)

    # Finally, we add our readout layer.
    out_width = structure['out']
    W = weight_variable([in_width, out_width])
    b = bias_variable([out_width])

    return tf.matmul(h_drop, W) + b, x, y, keep_prob
