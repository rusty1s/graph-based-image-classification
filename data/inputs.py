import os

import tensorflow as tf

from distort_image import (distort_image_for_train, distort_image_for_eval)

BATCH_SIZE = 128
MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
NUM_THREADS = 16


def inputs(dataset, batch_size=BATCH_SIZE, train=True, eval_data=False):
    """Constructs inputs from a dataset.

    Args:
        dataset: Instance of the dataset to use.
        batch_size: Number of data per batch.
        train: Boolean indiciating if one wants to train or evaluate. Default:
          True.
        eval_data: Boolean indicating if one should use the train or eval data
          set. Default: False.

    Returns:
        data_batch: 4D tensor of [batch_size, height, width, depth] size.
        label_batch: 1D tensor of [batch_size] size.
    """

    # When training, we want to apply a different distortion as if we are
    # evaluating.
    if train:
        distort = distort_image_for_train
        shuffle = True
    else:
        distort = distort_image_for_eval
        shuffle = False

    # Choose the right dataset.
    if not eval_data:
        filenames = dataset.train_filenames
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_train
    else:
        filenames = dataset.eval_filenames
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_eval

    # Construct the inputs.
    return _inputs(filenames, dataset.read, distort, num_examples_per_epoch,
                   batch_size, shuffle)


def _inputs(filenames, read, distort, num_examples_per_epoch, batch_size,
            shuffle):

    """Constructs inputs from data files using the read operation.

    Args:
        filenames: The filenames of the data batches in the data directory.
        read: Reader operation that returns a single record.
        distort: Distort operation on the data of the example.
        num_examples_per_epoch: Number of examples per epoch.
        batch_size: Number of data per batch.
        shuffle: Boolean indicating whether to use a shuffling queue.

    Returns:
        data_batch: 4D tensor of [batch_size, height, width, depth] size.
        label_batch: 1D tensor of [batch_size] size.
    """

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    record = read(filename_queue)

    # Distort the data.
    record = distort(record)

    min_queue_examples = int(num_examples_per_epoch *
                             MIN_FRACTION_OF_EXAMPLES_IN_QUEUE)
    capacity = min_queue_examples + 3 * batch_size

    print('Filling queue with {} examples before starting. This can take a '
          'few minutes.'.format(min_queue_examples))

    # Create a queue that shuffles the examples, and then read batch_size
    # data + labels from the example queue.
    if shuffle:
        data_batch, label_batch = tf.train.shuffle_batch(
            [record.data, record.label],
            batch_size=batch_size,
            num_threads=NUM_THREADS,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        data_batch, label_batch = tf.train.batch(
            [record.data, record.label],
            batch_size=batch_size,
            num_threads=NUM_THREADS,
            capacity=capacity)

    return data_batch, tf.reshape(label_batch, [batch_size])
