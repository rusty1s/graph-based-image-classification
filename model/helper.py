import tensorflow as tf


# To create the model, we're going to need to create a lot of weights and
# biases. One should generally initialize weights with a small amount of noise
# for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU
# neurons, it is also good practice to initialize them with a slightly positive
# initial bias to avoid "dead neurons".
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
