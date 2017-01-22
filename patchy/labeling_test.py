import tensorflow as tf

from .labeling import identity, betweenness_centrality, weight_to_first_label


class LabelingTest(tf.test.TestCase):

    def test_identity(self):
        adjacency = tf.constant([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ])

        expected = [0, 1, 2, 3]

        with self.test_session() as sess:
            labeling = identity(adjacency)
            self.assertAllEqual(labeling.eval(), expected)

    def test_betweenness_centrality(self):
        adjacency = tf.constant([
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
        ])

        labels = tf.constant([1, 2, 3, 4, 5, 6, 7])

        with self.test_session() as sess:
            labeling = betweenness_centrality(adjacency, labels)
            print(labeling.eval())

    def test_weight_to_first_label(self):
        pass
