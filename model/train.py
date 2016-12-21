# TODO das dataset als parameter h

import tensorflow as tf

from data import (Cifar10, inputs)
# from .helper import (weight_variable, bias_variable)
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


def _weight_variable(name, shape, stddev, decay):
    var = tf.get_variable(name, shape,
                          initializer=tf.truncated_normal_initializer(
                              stddev=stddev, dtype=tf.float32),
                          dtype=tf.float32)

    weight_decay = tf.mul(tf.nn.l2_loss(var), decay, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def _bias_variable(name, shape, constant):
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(constant),
                           dtype=tf.float32)


# {
#   conv: [
#      {
#        output_channels: 64
#        weights: { stddev, decay }
#        biases: { constant: 0.1 },
#        fields: { size: [5, 5], strides: [1, 1] }
#        max_pool: { size: [3, 3], strides: [2, 2] }
#     }
#   ],
#   local: [
#      {
#        output_channels: 1024,
#        weights: { stddev, decay }
#        biases: { constant: 0.1 },
#     }
#   ],
#   softmax_linear: {
#      output_channels: 10,
#      weights: { stddev, decay }
#      biases: { constant: 0.1 },
#   }
# }

def inference(data, structure):
    output = data
    i = 1

    for layer in structure['conv']:
        input_channels = output.get_shape()[3].value
        output_channels = layer['output_channels']

        weights_shape = (layer['fields']['size'] + [input_channels] +
                         [output_channels])

        strides = [1] + layer['fields']['strides'] + [1]

        with tf.variable_scope('conv_{}'.format(i)) as scope:

            weights = _weight_variable(
                name='weights',
                shape=weights_shape,
                stddev=layer['weights']['stddev'],
                decay=layer['weights']['decay'])

            biases = _bias_variable(
                name='biases',
                shape=[output_channels],
                constant=layer['biases']['constant'])

            output = tf.nn.conv2d(output, weights, strides, padding='SAME')
            output = tf.nn.bias_add(output, biases)
            output = tf.nn.relu(output, name=scope.name)

        max_pool_size = [1] + layer['max_pool']['size'] + [1]
        max_pool_strides = [1] + layer['max_pool']['strides'] + [1]

        output = tf.nn.max_pool(output, max_pool_size, max_pool_strides,
                                padding='SAME', name='pool_{}'.format(i))

        i += 1

    output = tf.reshape(output, [output.get_shape()[0].value, -1])

    for layer in structure['local']:
        input_channels = output.get_shape()[1].value
        output_channels = layer['output_channels']

        with tf.variable_scope('local_{}'.format(i)) as scope:

            weights = _weight_variable(
                name='weights',
                shape=[input_channels, output_channels],
                stddev=layer['weights']['stddev'],
                decay=layer['weights']['decay'])

            biases = _bias_variable(
                name='biases',
                shape=[output_channels],
                constant=layer['biases']['constant'])

            output = tf.matmul(output, weights) + biases
            output = tf.nn.relu(output, name=scope.name)

        i += 1

    layer = structure['softmax_linear']
    input_channels = output.get_shape()[1].value
    output_channels = layer['output_channels']

    with tf.variable_scope('softmax_linear') as scope:

        weights = _weight_variable(
            name='weights',
            shape=[input_channels, output_channels],
            stddev=layer['weights']['stddev'],
            decay=layer['weights']['decay'])

        biases = _bias_variable(
            name='biases',
            shape=[output_channels],
            constant=layer['biases']['constant'])

        output = tf.matmul(output, weights)
        output = tf.add(output, biases, name=scope.name)

    return output


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
