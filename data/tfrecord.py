import sys
import time

import tensorflow as tf
from six.moves import xrange

EPOCHS = 1
BATCH_SIZE = 100
NUM_THREADS = 16
CAPACITY = 1000


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write(dataset, filename, train=True, eval_data=False, epochs=EPOCHS,
          batch_size=BATCH_SIZE):

    if not eval_data:
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_train
        filenames = dataset.train_filenames
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch_for_eval
        filenames = dataset.eval_filenames

    steps = -(-num_examples_per_epoch // batch_size) * epochs

    filename_queue = tf.train.string_input_producer(filenames)

    record = dataset.read(filename_queue)

    if train:
        record = dataset.train_preprocess(record)
    else:
        record = dataset.eval_preprocess(record)

    data = record.data
    label = record.label

    data_batch, label_batch = tf.train.batch(
        [data, label],
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        capacity=CAPACITY)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    writer = tf.python_io.TFRecordWriter(filename)

    start_time = time.time()

    try:
        for i in xrange(1, steps+1):
            data, labels = sess.run([data_batch, label_batch])

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(record.height),
                'width': _int64_feature(record.width),
                'depth': _int64_feature(record.depth),
                'labels': _bytes_feature(labels.tostring()),
                'data': _bytes_feature(data.tostring()),
            }))

            writer.write(example.SerializeToString())

            remaining = (steps - i) * ((time.time() - start_time) / i) / 60

            sys.stdout.write(' '.join([
                '\r>> Writing',
                '{}'.format(dataset.name),
                '{}'.format('eval' if eval_data else 'train'),
                'dataset to',
                '{}'.format(filename),
                '{:.1f}%'.format(100*i/steps),
                '-',
                '{:.1f} min remaining'.format(remaining),
            ]))
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass

    finally:
        coord.request_stop()
        coord.join(threads)

        writer.close()
        sess.close()

        print()
        print(' '.join([
            'Successfully written {} batches'.format(i),
            '({:.2f} epochs)'.format(i * batch_size / num_examples_per_epoch),
        ]))
