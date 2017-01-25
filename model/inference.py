import tensorflow as tf


def _weight_variable(name, shape, stddev, decay):
    var = tf.get_variable(name, shape,
                          initializer=tf.truncated_normal_initializer(
                              stddev=stddev, dtype=tf.float32),
                          dtype=tf.float32)

    weight_decay = tf.mul(tf.nn.l2_loss(var), decay, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def _bias_variable(name, shape, constant):
    var = tf.get_variable(name, shape,
                          initializer=tf.constant_initializer(constant),
                          dtype=tf.float32)

    return var


# {
#   conv: [
#      {
#        output_channels: 64,
#        weights: { stddev, decay },
#        biases: { constant: 0.1 },
#        fields: { size: [5, 5], strides: [1, 1] },
#        max_pool: { size: [3, 3], strides: [2, 2] },
#     }
#   ],
#   local: [
#      {
#        output_channels: 1024,
#        weights: { stddev, decay },
#        biases: { constant: 0.1 },
#     }
#   ],
#   softmax_linear: {
#      output_channels: 10,
#      weights: { stddev, decay },
#      biases: { constant: 0.1 },
#   }
# }
def inference(data, network):
    output = data
    i = 1

    for layer in network['conv']:
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

    for layer in network['local']:
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

    layer = network['softmax_linear']
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
