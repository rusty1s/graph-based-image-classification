import tensorflow as tf


class NodeSequenceTest(tf.test.TestCase):

    def test_node_sequence(self):
        neighborhood = tf.constant([
            [1, 0, 3, -1],
            [2, 1, 0, -1],
        ])

        nodes = tf.constant([
            [0.5, 0.5, 0.5],
            [1.5, 1.5, 1.5],
            [2.5, 2.5, 2.5],
            [3.5, 3.5, 3.5],
        ])

        expected = [
            [[1.5, 1.5, 1.5], [0.5, 0.5, 0.5], [3.5, 3.5, 3.5], [0, 0, 0]],
            [[2.5, 2.5, 2.5], [1.5, 1.5, 1.5], [0.5, 0.5, 0.5], [0, 0, 0]],
        ]

        def _map_features(node):
            i = tf.maximum(node, 0)
            positive = tf.strided_slice(nodes, [i], [i+1], [1])
            negative = tf.zeros([1, 3])

            return tf.where(node < 0, negative, positive)

        with self.test_session() as sess:
            data = tf.reshape(neighborhood, [-1])
            data = tf.map_fn(_map_features, data, dtype=tf.float32)
            data = tf.reshape(data, [2, 4, 3])

            self.assertAllEqual(data.eval(), expected)
