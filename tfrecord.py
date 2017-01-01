import sys

import tensorflow as tf
from six.moves import xrange

from data import Cifar10DataSet

EPOCHS = 1
BATCH_SIZE = 100
NUM_THREADS = 16
CAPACITY = 1000


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write(dataset, train=True, epochs=EPOCHS, batch_size=BATCH_SIZE):
    steps = -(-dataset.num_examples_per_epoch_for_train // batch_size) * epochs

    filenames = dataset.train_filenames

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = dataset.read(filename_queue)
    data = read_input.data

    # TODO perform transformations here

    data_batch, label_batch = tf.train.batch(
        [data, read_input.label],
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        capacity=CAPACITY)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    filename = '/tmp/train.tfrecords'

    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.python_io.TFRecordWriter(filename)

        for i in xrange(1, steps+1):
            data, labels = sess.run([data_batch, label_batch])

            # TODO save in batches instead of this shitty for loop
            for j in xrange(BATCH_SIZE):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(read_input.height),
                    'width': _int64_feature(read_input.width),
                    'depth': _int64_feature(read_input.depth),
                    'label': _int64_feature(int(labels[j])),
                    'data': _bytes_feature(data[j].tostring()),
                }))

                writer.write(example.SerializeToString())

            sys.stdout.write(
                '\r>> Saving dataset to {}... {:.1f}%'
                .format(filename, 100*i/steps))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        coord.request_stop()
        coord.join(threads)

        writer.close()
        sess.close()


if __name__ == '__main__':
    dataset = Cifar10DataSet(data_dir='/tmp/cifar10_data')
    write(
        dataset=dataset,
        epochs=1,
        batch_size=100)
