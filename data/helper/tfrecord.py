import tensorflow as tf
import numpy as np

from .record import Record


def read_tfrecord(filename_queue, shapes={}):
    """Reads and parses TFRecord examples from data files.

    Args:
        filename_queue: A queue of strings with the filenames to read from.
        shapes: A dictionary containing the shape for a feature in a single
          example.

    Returns:
        A record object.
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = {key: tf.FixedLenFeature([], tf.string) for key in shapes}
    features['label'] = tf.FixedLenFeature([], tf.int64)

    example = tf.parse_single_example(serialized_example, features=features)

    data = {key: tf.decode_raw(example[key], tf.float32) for key in shapes}
    data = {key: tf.reshape(data[key], shapes[key]) for key in shapes}

    label = tf.reshape(example['label'], [1])

    return data, label


def write_tfrecord(writer, data, label):
    """Writes the data and label as a TFRecord example.

    Args:
        writer: A TFRecordReader.
        data: A dictionary holding numpy arrays of data.
        label: An int64 label index.
    """

    features = {key: _bytes_feature(data[key]) for key in data}
    features['label'] = _int64_feature(label)

    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(example.SerializeToString())


def _int64_feature(value):
    """Creates an int64 feature from the passed value.

    Args:
        value: An integer value.

    Returns:
        A TensorFlow feature.
    """

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _bytes_feature(value):
    """Creates a bytes feature from the passed value.

    Args:
        value: An numpy array.

    Returns:
        A TensorFlow feature.
    """

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value.astype(np.float32).tostring()]))
