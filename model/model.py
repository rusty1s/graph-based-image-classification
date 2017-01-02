import tensorflow as tf


def train_step(loss, step, learning_rate=0.1, epsilon=1.0):
    loss_averages_op = _add_loss_summaries(loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=epsilon)
        grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=step)

    variable_averages = tf.train.ExponentialMovingAverage(0.99999, step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])
    return loss_averages_op


def cal_loss(logits, labels):
    cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy = tf.reduce_mean(
        cross_entropy_per_example, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def cal_acc(logits, labels):
    corr_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(corr_prediction, tf.float32))

    return accuracy
