import tensorflow as tf


MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.


def train_step(loss, step, learning_rate, beta1, beta2, epsilon):
    loss_averages_op = _add_loss_summaries(loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name, var)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')

    # Attach a scalar summary to all individual losses and the total loss; do
    # the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as `(raw)` and name the moving average version of the
        # loss as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages.apply(losses + [total_loss])


def cal_loss(logits, labels):
    cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy = tf.reduce_mean(
        cross_entropy_per_example, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def cal_accuracy(logits, labels):
    corr_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(corr_prediction, tf.float32))

    # Add summary of the training accuracies.
    tf.summary.scalar('train_accuracy', accuracy)

    return accuracy
