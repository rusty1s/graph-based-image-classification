import tensorflow as tf
from skimage import io
from skimage import draw
import numpy as np

from data import Cifar10DataSet
from data import inputs

from grapher import SuperpixelGrapher
from superpixel.algorithm import slic_generator

slic = slic_generator(100)
grapher = SuperpixelGrapher(slic)


def main():
    cifar10 = Cifar10DataSet()
    data_batch, _ = inputs(cifar10, batch_size=1, train=False, eval_data=False)
    data_batch = tf.reshape(data_batch, [24, 24, 3])

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        data = sess.run(data_batch)
        data_op = tf.constant(data, tf.float32)
        segmentation = slic(data_op)
        segmentation = tf.reshape(segmentation, [-1])
        nodes, adjacent = grapher.create_graph(data_op)
        means = tf.strided_slice(nodes, [0, 0], [200, 3], [1, 1])
        data_op = tf.cast(data_op, tf.int32)
        centers = tf.strided_slice(nodes, [0, 3], [200, 5], [1, 1])
        centers = tf.cast(centers, tf.int32)

        def map(value):
            color = tf.strided_slice(means, [value], [value+1], [1])
            return tf.reshape(color, [3])

        image_op = tf.map_fn(map, segmentation, dtype=tf.float32)
        image_op = tf.reshape(image_op, [24, 24, 3])
        image_op = tf.cast(image_op, tf.int32)
        image = sess.run(image_op)
        centers = sess.run(centers)

        for center in centers:
            draw.set_color(image, center, [255, 0, 0])

        io.imsave('/home/vagrant/shared/org.png', data.astype(np.int32))
        io.imsave('/home/vagrant/shared/test.png', image)
        # print(image)
        print(centers)





    coord.request_stop()
    coord.join(threads)

    sess.close()


if __name__ == '__main__':
    main()
