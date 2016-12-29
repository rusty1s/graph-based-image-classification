import tensorflow as tf

from superpixel import create_graph


class GraphTest(tf.test.TestCase):

    def test_create_graph(self):
        image = tf.constant([
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        ])

        superpixels = tf.constant([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ])

        expected_nodes = [
            [1.5, 1.5, 1.5, 4, 0.25, 0.25, 2, 2],
            [3.5, 3.5, 3.5, 4, 0.25, 0.25, 2, 2],
            [1.5, 1.5, 1.5, 4, 0.25, 0.25, 2, 2],
            [3.5, 3.5, 3.5, 4, 0.25, 0.25, 2, 2],
        ]

        expected_adjacent = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

        with self.test_session() as sess:
            nodes, adjacent = create_graph(image, superpixels)
            self.assertAllEqual(nodes.eval(), expected_nodes)
            self.assertAllEqual(adjacent.eval(), expected_adjacent)
