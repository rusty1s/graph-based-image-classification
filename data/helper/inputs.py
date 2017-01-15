import os

import tensorflow as tf

BATCH_SIZE = 128
MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
NUM_THREADS = 16


def inputs(dataset, eval_data, batch_size=BATCH_SIZE, distort_inputs=True,
           num_epochs=None, shuffle=True):
    """Constructs inputs from a dataset.

    Args:
        dataset: Instance of the dataset to use.
        eval_data: Boolean indicating if one should use the train or eval data
          set. Default: False.
        batch_size: Number of data per batch (optional).
        distort_inputs: Boolean whether to distort the inputs (optional).
        num_epochs: Number indicating the maximal number of epochs iterations
          before raising an OutOfRange error (optional).
        shuffle: Boolean indiciating if one wants to shuffle the inputs
          (optional).

    Returns:
        data_batch: 4D tensor of [batch_size, height, width, depth] size.
        label_batch: 1D tensor of [batch_size] size.
    """

    # When shuffling one wants to also apply a different distortion.
    if shuffle:
        distort = dataset.distort_for_train
    else:
        distort = dataset.distort_for_eval

    # Choose the right dataset.
    if not eval_data:
        filenames = dataset.train_filenames
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_train
    else:
        filenames = dataset.eval_filenames
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_eval

    if num_epochs is None:
        filename_queue = tf.train.string_input_producer(
            filenames, shuffle=shuffle)
    else:
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs, shuffle)

    # Read examples from files in the filename queue.
    record = dataset.read(filename_queue)

    # Distort the data.
    if distort_inputs:
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
            capacity=capacity,
            allow_smaller_final_batch=False if num_epochs is None else True)

    return data_batch, tf.reshape(label_batch, [-1])
