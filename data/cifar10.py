import os

import tensorflow as tf

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


class Cifar10(DataSet):

    def __init__(self, data_dir='/tmp/cifar10_data'):
        maybe_download_and_extract(DATA_URL, data_dir)

        self._data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def train_filenames(self):
        return ['data_batch_{}.bin'.format(i) for i in range(1, 6)]

    @property
    def eval_filenames(self):
        return ['test_batch.bin']

    @property
    def num_labels(self):
        return 10

    @property
    def num_examples_per_epoch_for_train(self):
        return 50000

    @property
    def num_examples_per_epoch_for_eval(self):
        return 10000

    @property
    def data_shape(self):
        return [POST_HEIGHT, POST_WIDTH, 3]

    def read(self, filename_queue):
        """Reads and parses examples from CIFAR-10 data files.

        Args:
            filename_queue: A queue of strings with the filenames to read from.

        Returns:
            An object representing a single example, with the following fields:

            height: Number of rows in the example (32).
            width: Number of columns in the example (32).
            depth: Number of color channels in the example (3).
            key: a scalar string tensor describing the filename & record number
              for the example.
            label: An int32 tensor with the label of the example in the range
              0..9.
            data: A [height, width, depth] float32 tensor with the image data
              of the example.
        """

        class Record(object):
            pass

        result = Record()

        result.height = HEIGHT
        result.width = WIDTH
        result.depth = DEPTH

        # Read a record, getting filenames from the filename_queue. No header
        # or footer in the CIFAR-10 format, so we leave header_bytes and
        # footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=RECORD_BYTES)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is RECORD_BYTES long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8 to
        # int32.
        label = tf.strided_slice(record_bytes, [0], [LABEL_BYTES], [1])
        label = tf.cast(label, tf.int32)

        result.label = label

        # The reamining bytes after the label represent the image, which we
        # reshape from [depth * height * width] to [depth, height, width].
        data = tf.strided_slice(
            record_bytes, [LABEL_BYTES], [RECORD_BYTES], [1])
        data = tf.reshape(data, [DEPTH, HEIGHT, WIDTH])

        # Convert from [depth, height, width] to [height, width, depth].
        data = tf.transpose(data, [1, 2, 0])

        # Convert from uint8 to float32.
        data = tf.cast(data, tf.float32)

        result.data = data

        return result

    def train_preprocess(self, image):
        """Image processing for training the network with many random
        distortions applied to the image."""

        # Randomly crop a [height, width] section of the image.
        image = tf.random_crop(image, [POST_HEIGHT, POST_WIDTH, DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        return image

    def eval_preprocess(self, image):
        """Image processing for evaluating the network."""

        # Crop the central [height, width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(image, POST_HEIGHT,
                                                       POST_WIDTH)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        return image
