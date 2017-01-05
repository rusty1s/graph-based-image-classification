import os

import tensorflow as tf

from .record import Record
from .dataset import DataSet
from .download import maybe_download_and_extract

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Dimensions of the images in the CIFAR-10 dataset.
# See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the input
# format.
HEIGHT = 32
WIDTH = 32
DEPTH = 3

# Every record consists of a label followed by the image, with a fixed number
# of bytes for each.
LABEL_BYTES = 1
IMAGE_BYTES = HEIGHT * WIDTH * DEPTH
RECORD_BYTES = LABEL_BYTES + IMAGE_BYTES

# After preprocessing the image, its width and height will change.
POST_HEIGHT = 24
POST_WIDTH = 24


class Cifar10DataSet(DataSet):

    def __init__(self, data_dir='/tmp/cifar10_data'):
        maybe_download_and_extract(DATA_URL, data_dir)

        self._data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    @property
    def name(self):
        return 'cifar_10'

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def train_filenames(self):
        return tf.train.match_filenames_once(
            '{}/data_batch_*.bin'.format(self.data_dir))

    @property
    def eval_filenames(self):
        return tf.train.match_filenames_once(
            '{}/test_batch.bin'.format(self.data_dir))

    @property
    def num_labels(self):
        return 10

    @property
    def num_examples_per_epoch_for_train(self):
        return 50000

    @property
    def num_examples_per_epoch_for_eval(self):
        return 10000

    def read(self, filename_queue):
        """Reads and parses examples from CIFAR-10 data files.

        Args:
            filename_queue: A queue of strings with the filenames to read from.

        Returns:
            A record object.
        """

        # Read a record, getting filenames from the filename_queue. No header
        # or footer in the CIFAR-10 format, so we leave header_bytes and
        # footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=RECORD_BYTES)
        _, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is RECORD_BYTES long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        with tf.name_scope('read_label', values=[record_bytes]):
            # The first bytes represent the label, which we convert from uint8
            # to int64.
            label = tf.strided_slice(record_bytes, [0], [LABEL_BYTES], [1])
            label = tf.cast(label, tf.int64)

        with tf.name_scope('read_image', values=[record_bytes]):
            # The reamining bytes after the label represent the image, which we
            # reshape from [depth * height * width] to [depth, height, width].
            image = tf.strided_slice(
                record_bytes, [LABEL_BYTES], [RECORD_BYTES], [1])
            image = tf.reshape(image, [DEPTH, HEIGHT, WIDTH])

            # Convert from [depth, height, width] to [height, width, depth].
            image = tf.transpose(image, [1, 2, 0])

            # Convert from uint8 to float32.
            image = tf.cast(data, tf.float32)

        return Record(HEIGHT, WIDTH, DEPTH, label, image)

    def train_preprocess(self, record):
        """Image processing for training the network with many random
        distortions applied to the image."""

        image = record.data

        with tf.name_scope('train_distorted_input', values=[image]):
            # We need to convert the image to integer values, so the
            # distortions doesn't mess up with our image.
            image = tf.cast(image, tf.int32)

            # Randomly crop a [height, width] section of the image.
            image = tf.random_crop(image, [POST_HEIGHT, POST_WIDTH, DEPTH])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Ramdomly adjust the contrast of the image.
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

            # Convert the image back to the default type.
            image = tf.cast(image, tf.float32)

        # Return a new record with the applied distortions.
        return Record(POST_HEIGHT, POST_WIDTH, DEPTH, record.label, image)

    def eval_preprocess(self, record):
        """Image processing for evaluating the network."""

        image = record.data

        with tf.name_scope('eval_distorted_input', values=[image]):
            # Crop the central [height, width] of the image.
            image = tf.image.resize_image_with_crop_or_pad(image, POST_HEIGHT,
                                                           POST_WIDTH)

        # Return a new record with the applied distortions.
        return Record(POST_HEIGHT, POST_WIDTH, DEPTH, record.label, image)
