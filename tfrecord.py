import tensorflow as tf

from data import (write, read_and_decode)
from data import Cifar10DataSet



def main():

    return data_batch, label_batch


if __name__ == '__main__':
    dataset = Cifar10DataSet(data_dir='/tmp/cifar10_data')
    filename = '/tmp/citrain.tfrecords'

    write(
        dataset=dataset,
        filename=filename,
        train=False,
        eval_data=True,
        epochs=1,
        batch_size=100)

    filename = '/tmp/citrain.tfrecords'
    filename_queue = tf.train.string_input_producer([filename])

    record = read_and_decode(filename_queue, 24, 24, 3)

    data_batch, label_batch = tf.train.shuffle_batch(
        [record.data, record.label],
        batch_size=128,
        num_threads=2,
        capacity=1300,
        min_after_dequeue=1000)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        print(i)
        labels = sess.run(label_batch)
        print(labels)

    coord.request_stop()
    coord.join(threads)

    sess.close()
