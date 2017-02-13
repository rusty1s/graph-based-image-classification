import tensorflow as tf

from .labeling import scanline, betweenness_centrality, canonical


class LabelingTest(tf.test.TestCase):

    def test_scanline(self):
        adjacency = tf.constant([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ])

        expected = [0, 1, 2, 3]

        with self.test_session() as sess:
            labeling = scanline(adjacency)
            self.assertAllEqual(labeling.eval(), expected)

    def test_betweenness_centrality(self):
        adjacency = tf.constant([
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        labels = tf.constant([1, 2, 3, 4, 5, 6, 7])

        expected = [4, 3, 2, 6, 5, 1, 7]

        with self.test_session() as sess:
            labeling = betweenness_centrality(adjacency, labels)
            self.assertAllEqual(labeling.eval(), expected)

    def test_canonical(self):
        adjacency = tf.constant([
            [0, 1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
        ])

        labels = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])

        expected = [4, 7, 5, 8, 6, 2, 1, 3]

        with self.test_session() as sess:
            labeling = canonical(adjacency, labels)
            self.assertAllEqual(labeling.eval(), expected)

        # Create isomorph graph and check for equal canonical labeling.
        # adjacency = tf.constant([
        #     [0, 1, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 1, 1, 0, 0, 1, 0],
        #     [0, 1, 0, 1, 1, 0, 1, 0],
        #     [0, 1, 1, 0, 0, 1, 1, 0],
        #     [0, 0, 1, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 1, 1, 0, 0, 0],
        #     [0, 1, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 1, 0],
        # ])

        # with self.test_session() as sess:
        #     labeling = canonical(adjacency, labels)
        #     self.assertAllEqual(labeling.eval(), expected)
