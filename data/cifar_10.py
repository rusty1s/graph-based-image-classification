import os

import tensorflow as tf

from .dataset import DataSet
from .helper.record import Record
from .helper.download import maybe_download_and_extract
from .helper.distort_image import distort_image_for_train,\
                                  distort_image_for_eval


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_DIR = '/tmp/cifar_10_data'

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

# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


class Cifar10(DataSet):
    """CIFAR-10 dataset."""

    def __init__(self, data_dir=DATA_DIR, show_progress=True):
        """Creates a CIFAR-10 dataset.

        Args:
            data_dir: The path to the directory where the CIFAR-10 dataset is
            stored.
            show_progress: Show a pretty progress bar for dataset computations.
        """

        super().__init__(data_dir, show_progress)

        maybe_download_and_extract(DATA_URL, data_dir, show_progress)

    @property
    def train_filenames(self):
        """The filenames of the training batches from the CIFAR-10 dataset."""

        return tf.train.match_filenames_once(
            os.path.join(self.data_dir, 'cifar-10-batches-bin',
                         'data_batch_*.bin'))

    @property
    def eval_filenames(self):
        """The filenames of the evaluation batches from the CIFAR-10
        dataset."""

        return [os.path.join(self.data_dir, 'cifar-10-batches-bin',
                             'test_batch.bin')]

    @property
    def labels(self):
        """The ordered labels of the CIFAR-10 dataset."""

        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck']

    @property
    def num_examples_per_epoch_for_train(self):
        """The number of examples per epoch for training the CIFAR-10 dataset.
        """

        return NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    @property
    def num_examples_per_epoch_for_eval(self):
        """The number of examples per epoch for evaluating the CIFAR-10
        dataset."""

        return NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    def read(self, filename_queue):
        """Reads and parses examples from CIFAR-10 data files."""

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
            image = tf.cast(image, tf.float32)

        return Record(image, [HEIGHT, WIDTH, DEPTH], label)

    def distort_for_train(self, record):
        """Applies random distortions for training to a CIFAR-10 record."""

        return distort_image_for_train(record)

    def distort_for_eval(self, record):
        """Applies distortions for evaluation to a CIFAR-10 record."""

        return distort_image_for_eval(record)
