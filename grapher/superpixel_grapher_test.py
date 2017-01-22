import tensorflow as tf

from segmentation.algorithm import slico_generator

from .superpixel_grapher import SuperpixelGrapher


class SuperpixelGrapherTest(tf.test.TestCase):

    def test_create_graph(self):
        image = tf.constant([
            [[255, 255, 255], [255, 255, 255], [1, 1, 1], [1, 1, 1]],
            [[255, 255, 255], [255, 255, 255], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [254, 254, 254], [254, 254, 254]],
            [[0, 0, 0], [0, 0, 0], [254, 254, 254], [254, 254, 254]],
        ], dtype=tf.float32)

        expected_nodes = [
            [255, 255, 255, 0.5, 0.5, 4, 2, 2],
            [1, 1, 1, 0.5, 0.5, 4, 2, 2],
            [0, 0, 0, 0.5, 0.5, 4, 2, 2],
            [254, 254, 254, 0.5, 0.5, 4, 2, 2],
        ]

        expected_adjacent = [
            [0, 0, 0, 3],
            [0, 0, 3, 0],
            [0, 3, 0, 0],
            [3, 0, 0, 0],
        ]

        slico = slico_generator(num_segments=4)
        grapher = SuperpixelGrapher(slico)

        with self.test_session() as sess:
            nodes, adjacent = grapher.create_graph(image)
            self.assertAllEqual(nodes.eval(), expected_nodes)
            self.assertAllEqual(adjacent.eval(), expected_adjacent)
