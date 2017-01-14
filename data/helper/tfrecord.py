import tensorflow as tf
import numpy as np

from .record import Record


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
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, shape)

    label = tf.reshape(features['label'], [1])

    return Record(data, shape, label)


def write_to_tfrecord(writer, data, label):
    """Writes the data and label as a TFRecord example.

    Args:
        writer: A TFRecordReader.
        data: A numpy array holding the data.
        label: An int64 label index.
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'data': _bytes_feature(data.astype(np.float32)),
        'label': _int64_feature(label),
    }))

    writer.write(example.SerializeToString())


def _int64_feature(value):
    """Creates an int64 feature from the passed value.

    Args:
        value: An integer value.

    Returns:
        A TensorFlow feature.
    """

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Creates a bytes feature from the passed value.

    Args:
        value: An numpy array.

    Returns:
        A TensorFlow feature.
    """

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.tostring()]))
