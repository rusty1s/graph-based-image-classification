import tensorflow as tf

from .patchy_san import betweenness_centrality
from .patchy_san import node_sequence


class PatchySanTest(tf.test.TestCase):

    def test_betweenness_centrality(self):
        adjacent = tf.constant([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0],
        ])

        expected = [2, 1, 0, 3, 4, 5]

        with self.test_session() as sess:
            labeling = betweenness_centrality(adjacent)
            self.assertAllEqual(labeling.eval(), expected)

    def test_node_sequence(self):
        sequence = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])

        with self.test_session() as sess:
            s = node_sequence(sequence, width=4, stride=1)
            self.assertAllEqual(s.eval(), [1, 2, 3, 4])

            s = node_sequence(sequence, width=10, stride=1)
            self.assertAllEqual(s.eval(), [1, 2, 3, 4, 5, 6, 7, 8, 0, 0])

            s = node_sequence(sequence, width=4, stride=2)
            self.assertAllEqual(s.eval(), [1, 3, 5, 7])

            s = node_sequence(sequence, width=5, stride=2)
            self.assertAllEqual(s.eval(), [1, 3, 5, 7, 0])

            s = node_sequence(sequence, width=5, stride=3)
            self.assertAllEqual(s.eval(), [1, 4, 7, 0, 0])
