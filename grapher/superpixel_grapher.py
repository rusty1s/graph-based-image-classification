import tensorflow as tf

from .grapher import Grapher


class SuperpixelGrapher(Grapher):

    def __init__(self, superpixel_algorithm):
        self._superpixel_algorithm = superpixel_algorithm

    @property
    def num_node_channels(self):
        return 8

    def create_graph(self, image):
        image = tf.cast(image, tf.float32)
        segmentation = self._superpixel_algorithm(image)
        num_segments = tf.reduce_max(segmentation) + 1
        values = tf.range(0, num_segments, dtype=tf.int32)

        ones = tf.ones_like(segmentation)

        def _bounding_box(mask):
            cols = tf.reduce_any(mask, axis=0)
            cols = tf.where(cols)
            cols = tf.reshape(cols, [-1])
            rows = tf.reduce_any(mask, axis=1)
            rows = tf.where(rows)
            rows = tf.reshape(rows, [-1])
            top = tf.strided_slice(rows, [0], [1], [1])
            bottom = tf.strided_slice(tf.reverse(rows, [True]), [0], [1], [1]) + 1
            left = tf.strided_slice(cols, [0], [1], [1])
            right = tf.strided_slice(tf.reverse(cols, [True]), [0], [1], [1]) + 1
            return top, right, bottom, left

        def _extract(value):
            mask = tf.equal(segmentation, value * ones)

            # get the bounding box
            top, right, bottom, left = _bounding_box(mask)

            # slice the mask to the bounding box
            mask = tf.strided_slice(mask, tf.concat(0, [top, left]),
                                    tf.concat(0, [bottom, right]), [1, 1])

            # calculate center
            indices = tf.where(mask)
            indices = tf.transpose(indices)
            indices = tf.cast(indices, tf.float32)
            center = tf.reduce_mean(indices, axis=1)

            # calculate count
            count = tf.count_nonzero(mask, dtype=tf.float32)

            # calculate mean color
            img = tf.strided_slice(image, tf.concat(0, [top, left]),
                                   tf.concat(0, [bottom, right]), [1, 1])
            img = tf.reshape(img, [-1, 3])
            img = tf.select(tf.reshape(mask, [-1]), img, tf.zeros_like(img))
            color_sum = tf.reduce_sum(img, 0)
            mean = tf.div(color_sum, count * tf.ones_like(color_sum))

            features = tf.concat(0, [
                tf.reshape(mean, [-1]),
                tf.reshape(center, [-1]),
                [count],
                tf.cast(right - left, tf.float32),
                tf.cast(bottom - top, tf.float32),
            ])
            return features

        nodes = tf.map_fn(_extract, values, dtype=tf.float32)

        # adjacent, difference in color values with threshold
        means = tf.strided_slice(nodes, [0, 0], [num_segments, 3], [1, 1])
        a = tf.expand_dims(means, 1)
        b = tf.expand_dims(means, 0)
        distances = tf.reduce_sum(tf.squared_difference(a, b), 2)

        # Apply threshold
        threshold = tf.ones_like(distances) * 20
        mask = tf.less(distances, threshold)
        distances = tf.select(mask, distances, tf.zeros_like(distances))

        return nodes, distances
