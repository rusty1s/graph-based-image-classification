import tensorflow as tf

from data import Cifar10DataSet
from data import PatchySanDataSet
from data import inputs


def main():
    cifar10 = Cifar10DataSet()
    patchy = PatchySanDataSet(dataset=cifar10)

    data_batch, label_batch = inputs(patchy, batch_size=1, train=False,
                                     eval_data=True)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        data, labels = sess.run([data_batch, label_batch])
        print(labels)
        print(data)

    coord.request_stop()
    coord.join(threads)

    sess.close()


if __name__ == '__main__':
    main()
