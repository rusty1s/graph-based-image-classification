import tensorflow as tf


def inputs(data_dir, filenames, read, preprocess, num_examples_per_epoch,
           batch_size, shuffle=True):

    """Constructs inputs using the Reader ops.

    Args:
        data_dir: Path to the data directory.
        filenames: The filenames of the data batches in the data directory.
        read: Reader operation that returns a single example, with the
          following fields:
            height: Number of rows in the example.
            width: Number of colums in the example.
            depth: Number of channels in the example.
            key: A scalar string Tensor describing the filename & record number
              for this example.
            label: an int32 Tensor with the label of the example in the range
              0..num_labels.
            data: A [height, width, depth] float32 Tensor with the data of the
              example.
        preprocess: Preprocess operation on the data of the example.
        num_examples_per_epoch: Number of examples per epoch.
        batch_size: Number of data per batch.
        shuffle: Boolean indicating whether to use a shuffling queue.

    Returns:
        data_batch: 4D tensor of [batch_size, height, width, depth] size.
        label_batch: 1D tensor of [batch_size] size.
    """

    filenames = [os.path.join(data_dir, f) for f in filenames]

    for f in filenames:
        if not tf.gFile.Exists(f):
            raise ValueError('Failed to find file: {}'.format(f))

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read(filename_queue)

    height = read_input.width
    width = read_input.height
    depth = read_input.depth

    # Preprocess the data.
    data = preprocess(read_input.data)

    # Set the shapes of tensors.
    data.set_shape([height, width, depth])
    read_input.labels.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with {} examples before starting. This can take a '
          'few minutes.'.format(min_queue_examples))

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_data_and_label_batch(data, read_input.label,
                                          min_queue_examples, batch_size,
                                          shuffle)


def _generate_image_and_label_batch(data, label, min_queue_examples,
                                    batch_size, shuffle):

    """Constructs a queued batch of data and labels.

    Args:
        data: 3D tensor of [height, width, depth] of type float32.
        label: 1D tensor of type int32.
        min_queue_examples: int32, minimum number of samples to retain in the
          queue that provides of batches of examples.
        batch_size: Number of data per batch.
        shuffle: Boolean indicating whether to use a shuffling queue.

    Returns:
        data_batch: 4D tensor of [batch_size, height, width, depth] size.
        label_batch: 1D tensor of [batch_size] size.
    """

    num_threads = 16

    # Create a queue that shuffles the examples, and then read `batch_size`
    # data + labels from the example queue.
    if shuffle:
        data_batch, label_batch = tf.train.shuffle_batch(
            [data, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
        )
    else:
        data_batch, label_batch = tf.train.batch(
            [data, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size,
        )

    return data_batch, tf.reshape(label_batch, [batch_size])
