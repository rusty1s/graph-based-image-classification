import tensorflow as tf

from .superpixel_grapher import SuperpixelGrapher
from .superpixel.algorithm import slico_generator


class SuperpixelGrapherTest(tf.test.TestCase):

    def test_create_graph(self):
        image = tf.constant([
            [255, 255, 0,   0],
            [255, 255, 0,   0],
            [0,   0,   255, 255],
            [0,   0,   255, 255],
        ])

        slico = slico_generator(num_superpixels=4)
        grapher = SuperpixelGrapher(slico)

        expected_nodes = [
            [255.0, 0, 0, 4, 0.25, 0.25, 2, 2],
            [0.0, 0.0, 0.0, 4, 0.25, 0.25, 2, 2],
            [0.0, 0.0, 0.0, 4, 0.25, 0.25, 2, 2],
            [255.0, 0.0, 0.0, 4, 0.25, 0.25, 2, 2],
        ]

        expected_adjacent = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

        with self.test_session() as sess:
            nodes, adjacent = grapher.create_graph(image)
            self.assertAllEqual(nodes.eval(), expected_nodes)
            self.assertAllEqual(adjacent.eval(), expected_adjacent)
