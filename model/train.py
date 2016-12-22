import json

import tensorflow as tf

from data import inputs
from .inference import inference
from .hooks import hooks

MOVING_AVERAGE_DECAY = 0.9999


def train_step(total_loss, global_step):
    lr = learning_rate(50000, 128, global_step, 350.0, 0.1, 0.1)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_step = tf.no_op(name='train')

    return train_step


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


def cal_loss(logits, labels):
    # labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def cal_acc(logits, labels):
    labels = tf.cast(labels, tf.int64)
    corr_pred = tf.equal(tf.argmax(logits, 1), labels)
    acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    return acc


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op


def train(dataset, train_dir, network_params_path):
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    with open(network_params_path, 'r') as f:
        network_params = json.load(f)

    structure = network_params['structure']
    batch_size = network_params['batch_size']
    last_step = network_params['last_step']

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        data, labels = inputs(dataset, batch_size)

        logits = inference(data, structure)

        loss = cal_loss(logits, labels)
        acc = cal_acc(logits, labels)

        op = train_step(loss, global_step)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_dir,
                save_checkpoint_secs=30,
                hooks=hooks(display_step=10, last_step=last_step,
                            batch_size=batch_size, loss=loss, accuracy=acc)
                ) as monitored_session:
            while not monitored_session.should_stop():
                monitored_session.run(op)
