import tensorflow as tf
# ich brauche inputs
# das dataset als parameter h

from data import (Cifar10, inputs)

cifar10 = Cifar10()


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

    tf.contrib.deprecated.scalar_summary('learning_rate', learning_rate)

    return learning_rate


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = inputs(cifar10, batch_size=128)

        print(labels)
        print(images)
