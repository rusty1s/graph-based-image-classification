import tensorflow as tf

from .patchy_san import betweenness_centrality


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
