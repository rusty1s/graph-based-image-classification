import os

import tensorflow as tf

from .record import Record

MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
NUM_THREADS = 16


def inputs(dataset, eval_data, batch_size=128, scale_inputs=1.0,
           distort_inputs=False, zero_mean_inputs=False, num_epochs=None,
           shuffle=False):
    """Constructs inputs from a dataset.

    Args:
        dataset: Instance of the dataset to use.
        eval_data: Boolean indicating if one should use the train or eval data
          set.
        batch_size: Number of data per batch (optional).
        scale_inputs: Float defining the scaling to use for resizing the
          record's data (optional).
        distort_inputs: Boolean whether to distort the inputs (optional).
        zero_mean_inputs: Boolean indicating if one should linearly scales the
          record's data to have zero mean and unit norm (optional).
        num_epochs: Number indicating the maximal number of epoch iterations
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

    # Choose the right dataset filenames.
    if not eval_data:
        if shuffle:
            filenames = dataset.train_filenames
            num_examples_per_epoch = dataset.num_examples_per_epoch_for_train
        else:
            filenames = dataset.train_eval_filenames
            num_examples_per_epoch =\
                dataset.num_examples_per_epoch_for_train_eval
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

    if scale_inputs != 1.0:
        record = _resize(record, scale=scale_inputs)

    if distort_inputs:
        record = distort(record)

    if zero_mean_inputs:
        record = _zero_mean(record)

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


def _resize(record, scale):
    """Resizes the record's data using area interpolation.

    Args:
        record: The record.
        scale: Float defining the scaling to use for resizing.

    Returns:
        A new record object after applying resizing.
    """

    new_height = int(scale * record.shape[0])
    new_width = int(scale * record.shape[1])

    with tf.name_scope('resize', values=[record.data, new_height, new_width]):
        data_batch = tf.expand_dims(record.data, axis=0)
        data_batch = tf.image.resize_area(
            data_batch, [new_height, new_width], align_corners=True)
        data = tf.squeeze(data_batch, axis=[0])

    return Record(data, [new_height, new_width, record.shape[2]], record.label)


def _zero_mean(record):
    """Linearly scales the record's data to have zero mean and unit norm.

    Args:
        record: The record.

    Returns:
        A new record object after applying standardization.
    """

    data = tf.image.per_image_standardization(record.data)
    return Record(data, record.shape, record.label)
