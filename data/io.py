import sys
import time

import tensorflow as tf
from six.moves import xrange

from .record import Record

EPOCHS = 1
BATCH_SIZE = 100
NUM_THREADS = 16
CAPACITY = 1000


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.tostring()]))


def get_example(data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'data': _bytes_feature(data),
    }))


def write(num_examples_per_epoch, input_filenames, output_filename, read,
          preprocess=None, epochs=EPOCHS, batch_size=BATCH_SIZE,
          eval_data=False, dataset_name='', show_progress=True):

    steps = -(-num_examples_per_epoch // batch_size) * epochs

    filename_queue = tf.train.string_input_producer(input_filenames)

    record = read(filename_queue)

    record = preprocess(record)

    data_batch, label_batch = tf.train.batch(
        [record.data, record.label],
        batch_size=batch_size,
        num_threads=NUM_THREADS,
        capacity=CAPACITY)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    writer = tf.python_io.TFRecordWriter(output_filename)

    start_time = time.time()

    try:
        for i in xrange(1, steps+1):
            data, labels = sess.run([data_batch, label_batch])

            for j in xrange(batch_size):
                example = get_example(data[j], int(labels[j][0]))
                writer.write(example.SerializeToString())

            remaining = (steps - i) * ((time.time() - start_time) / i) / 60

            if show_progress:
                sys.stdout.write(' '.join([
                    '\r>> Writing {}'.format(dataset_name),
                    '{}'.format('eval' if eval_data else 'train'),
                    'dataset to',
                    '{}'.format(output_filename),
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
            'Successfully written {} examples'.format(i*batch_size),
            '({:.2f} epochs).'.format(i * batch_size / num_examples_per_epoch),
        ]))


def read_and_decode(filename_queue, height, width, depth):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'data': tf.FixedLenFeature([], tf.string),
        })

    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, [height, width, depth])

    label = tf.reshape(features['label'], [1])

    return Record(height, width, depth, label, data)
