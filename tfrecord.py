import tensorflow as tf

from data import (write, read_and_decode)
from data import inputs
from data import Cifar10DataSet


if __name__ == '__main__':
    dataset = Cifar10DataSet(data_dir='/tmp/cifar10_data')
    data_batch, label_batch = inputs(dataset, batch_size=1, train=False, eval_data=True)
    # filename_queue = tf.train.string_input_producer([dataset.eval_filenames])
    # record = dataset.read(filename_queue)
    # print(record.data)
    # print(record.label)
    # filename = '/tmp/citrain.tfrecords'

    # write(10000, dataset.eval_filenames, filename, dataset.read,
    #       preprocess=dataset.eval_preprocess, epochs=1, batch_size=128,
    #       eval_data=True, dataset_name='CIFAR-10', show_progress=True)

    # filename_queue = tf.train.string_input_producer([filename])

    # record = read_and_decode(filename_queue, 24, 24, 3)

    # data_batch, label_batch = tf.train.shuffle_batch(
    #     [record.data, record.label],
    #     batch_size=1,
    #     num_threads=2,
    #     capacity=1300,
    #     min_after_dequeue=1000)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        images = sess.run(data_batch)
        print(images)

    coord.request_stop()
    coord.join(threads)

    sess.close()
