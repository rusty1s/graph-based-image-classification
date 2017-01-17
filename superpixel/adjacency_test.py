import tensorflow as tf

from .adjacency import mean_color_adjacency


class MeanColorAdjacencyTest(tf.test.TestCase):

    def test_adjacency(self):
        image = tf.constant([
            [[10, 0, 0], [10, 0, 0],  [20, 0, 0],  [20, 0, 0]],
            [[10, 0, 0], [10, 0, 0],  [20, 0, 0],  [20, 0, 0]],
            [[30, 0, 0], [30, 0, 0],  [40, 0, 0],  [40, 0, 0]],
            [[30, 0, 0], [30, 0, 0],  [40, 0, 0],  [40, 0, 0]],
        ], dtype=tf.float32)

        segmentation = tf.constant([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ], dtype=tf.int32)

        expected = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

        with self.test_session() as sess:
            neighbors, _ = mean_color_adjacency(image, segmentation)
            self.assertAllEqual(neighbors.eval(), expected)

    def test_mean_color(self):
        image = tf.constant([
            [[10, 0, 0], [10, 0, 0],  [20, 0, 0],  [20, 0, 0]],
            [[10, 0, 0], [10, 0, 0],  [20, 0, 0],  [20, 0, 0]],
            [[30, 0, 0], [30, 0, 0],  [40, 0, 0],  [40, 0, 0]],
            [[30, 0, 0], [30, 0, 0],  [40, 0, 0],  [40, 0, 0]],
        ], dtype=tf.float32)

        segmentation = tf.constant([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ], dtype=tf.int32)

        expected = [
            [0, 10, 20, 30],
            [10, 0, 10, 20],
            [20, 10, 0, 10],
            [30, 20, 10, 0],
        ]

        with self.test_session() as sess:
            _, mean_color = mean_color_adjacency(image, segmentation)
            self.assertAllEqual(mean_color.eval(), expected)
