import os

import tensorflow as tf

BATCH_SIZE = 128


def inputs(dataset, batch_size=BATCH_SIZE, train=True, eval_data=False):
    """ Constructs inputs using the Reader ops.

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

    # When training, we want to apply a different preprocessing step as if we
    # are evaluating.
    if train:
        preprocess = dataset.train_preprocess
        shuffle = True
    else:
        preprocess = dataset.eval_preprocess
        shuffle = False

    # Choose the right dataset.
    if not eval_data:
        filenames = dataset.train_filenames
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_train
    else:
        filenames = dataset.eval_filenames
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_eval

    # Construct the inputs.
    return _inputs(dataset.data_dir, filenames, dataset.read,
                   preprocess, num_examples_per_epoch, batch_size, shuffle)


def _inputs(data_dir, filenames, read, preprocess, num_examples_per_epoch,
            batch_size, shuffle):

    """Constructs inputs using the Reader ops.

    Args:
        data_dir: Path to the data directory.
        filenames: The filenames of the data batches in the data directory.
        read: Reader operation that returns a single example, with the
          following fields:
            height: Number of rows in the example.
            width: Number of colums in the example.
            depth: Number of channels in the example.
            label: An int32 tensor with the label of the example in the range
              0..num_labels.
            data: A [height, width, depth] float32 tensor with the data of the
              example.
        preprocess: Preprocess operation on the data of the example.
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

    # Preprocess the data.
    record = preprocess(record)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    min_queue_examples = 200

    print('Filling queue with {} examples before starting. This can take a '
          'few minutes.'.format(min_queue_examples))

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_data_and_label_batch(record.data, record.label,
                                          min_queue_examples, batch_size,
                                          shuffle)


def _generate_data_and_label_batch(data, label, min_queue_examples,
                                   batch_size, shuffle):

    """Constructs a queued batch of data and labels.

    Args:
        data: 3D tensor of [height, width, depth] of type float32.
        label: 1D tensor of type int32.
        min_queue_examples: Minimum number of samples to retain in the queue
          that provides of batches of examples.
        batch_size: Number of data per batch.
        shuffle: Boolean indicating whether to use a shuffling queue.

    Returns:
        data_batch: 4D tensor of [batch_size, height, width, depth] size.
        label_batch: 1D tensor of [batch_size] size.
    """

    num_threads = 16
    capacity = min_queue_examples + 3 * batch_size

    # Create a queue that shuffles the examples, and then read batch_size
    # data + labels from the example queue.
    if shuffle:
        data_batch, label_batch = tf.train.shuffle_batch(
            [data, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        data_batch, label_batch = tf.train.batch(
            [data, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity)

        # Display the data_batch in the visualizer.
        # tf.summary.image('images', data_batch)

    return data_batch, tf.reshape(label_batch, [batch_size])
