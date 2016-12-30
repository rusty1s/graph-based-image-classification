import tensorflow as tf

from superpixel.algorithm import slico_generator

from .superpixel_grapher import SuperpixelGrapher

import numpy as np


class SuperpixelGrapherTest(tf.test.TestCase):

    def test_create_graph(self):
        img1 = np.zeros((10, 10))
        img1[2:7, 2:7] = 1.0

        cols = np.any(img1, axis=0)
        rows = np.any(img1, axis=1)
        rows = np.where(rows)[0]
        # print(rows)



        image = tf.constant([
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        ], dtype=tf.float32)

        slico = slico_generator(num_superpixels=4)
        grapher = SuperpixelGrapher(slico)

        with self.test_session() as sess:
            # self.assertEqual(grapher.node_channels_length, 8)
            nodes = grapher.create_graph(image)
            print(nodes.eval())
            # self.assertAllEqual(nodes.eval(), expected_nodes)
            # self.assertAllEqual(adjacent.eval(), expected_adjacent)


    # def test_create_graph(self):
    #     image = tf.constant([
    #         [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
    #         [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
    #         [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
    #         [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
    #     ])

    #     slico = slico_generator(num_superpixels=4)
    #     grapher = SuperpixelGrapher(slico)

    #     expected_nodes = [
    #         [255.0, 255.0, 255.0, 4, 0.25, 0.25, 2, 2],
    #         [0.0, 0.0, 0.0, 4, 0.25, 0.25, 2, 2],
    #         [0.0, 0.0, 0.0, 4, 0.25, 0.25, 2, 2],
    #         [255.0, 255.0, 255.0, 4, 0.25, 0.25, 2, 2],
    #     ]

    #     expected_adjacent = [
    #         [0, 1, 1, 1],
    #         [1, 0, 1, 1],
    #         [1, 1, 0, 1],
    #         [1, 1, 1, 0],
    #     ]

    #     with self.test_session() as sess:
    #         self.assertEqual(grapher.node_channels_length, 8)
    #         nodes, adjacent = grapher.create_graph(image)
    #         self.assertAllEqual(nodes.eval(), expected_nodes)
    #         self.assertAllEqual(adjacent.eval(), expected_adjacent)
