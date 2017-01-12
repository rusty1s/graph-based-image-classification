import tensorflow as tf

from .record import Record


def tfrecord_example(data, label):
    """Converts the data and label to a TFRecord example.

    Args:
        data: A numpy array holding the data.
        label: An int64 label index.

    Returns:
        A TFRecord example.
    """

    return tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'data': _bytes_feature(data),
    }))


def read_tfrecord(filename_queue, shape):
    """Reads and parses TFRecord examples from data files.

    Args:
        filename_queue: A queue of strings with the filenames to read from.
        shape: A TensorShape representing the shape of the data.

    Returns:
        A record object.
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'data': tf.FixedLenFeature([], tf.string),
        })

    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, shape)

    label = tf.reshape(features['label'], [1])

    return Record(data, shape, label)


def _int64_feature(value):
    """Creates an int64 feature from the passed value."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Creates an bytes feature from the passed value."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.tostring()]))
