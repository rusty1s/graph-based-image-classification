import tensorflow as tf

from superpixel.algorithm import slico_generator

from .superpixel_grapher import SuperpixelGrapher


class SuperpixelGrapherTest(tf.test.TestCase):

    def test_mean_pair(self):
        A = tf.constant([[0, 0, 0], [5, 5, 5], [30, 30, 30]], dtype=tf.float32)
        expanded_a = tf.expand_dims(A, 1)
        expanded_b = tf.expand_dims(A, 0)
        distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)

        # A = tf.reduce_sum(A, 1)

        with self.test_session() as sess:
            pass
            # print('A', A.eval())
            # print('a', expanded_a.eval())
            # print('b', expanded_b.eval())
            # print('distances', distances.eval())
            # print('X', X.eval())
            # print('r', r.eval())
            # print('D', D.eval())


    def test_create_graph(self):
        image = tf.constant([
            [[255, 255, 255], [255, 255, 255], [1, 1, 1], [1, 1, 1]],
            [[255, 255, 255], [255, 255, 255], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [254, 254, 254], [254, 254, 254]],
            [[0, 0, 0], [0, 0, 0], [254, 254, 254], [254, 254, 254]],
        ], dtype=tf.float32)

        slico = slico_generator(num_superpixels=4)
        grapher = SuperpixelGrapher(slico)

        with self.test_session() as sess:
            # self.assertEqual(grapher.node_channels_length, 8)
            nodes, adjacent = grapher.create_graph(image)
            print(nodes.eval())
            print(adjacent.eval())
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
