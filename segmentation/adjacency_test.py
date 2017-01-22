import tensorflow as tf
import numpy as np

from .adjacency import adjacency, adjacency_euclid_distance


class AdjacencyTest(tf.test.TestCase):

    def test_adjacency(self):
        segmentation = tf.constant([
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [2, 0, 0, 3],
            [2, 2, 3, 3],
        ], dtype=tf.int32)

        expected = [
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
        ]

        with self.test_session() as sess:
            adjacency_matrix = adjacency(segmentation)
            self.assertAllEqual(adjacency_matrix.eval(), expected)

    def test_adjacency_euclid_distance(self):
        segmentation = tf.constant([
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [2, 0, 0, 3],
            [2, 2, 3, 3],
        ], dtype=tf.int32)

        c = np.array([
            [(0+1+0+1+2+1+2)/7, (0+1+0+1+2+1+2)/7],
            [(0+0+1)/3, (2+3+3)/3],
            [(2+3+3)/3, (0+0+1)/3],
            [(3+2+3)/3, (3+2+3)/3],
        ], dtype=np.float32)

        def _euclid(centroid, index1, index2):
            return np.linalg.norm(centroid[index1] - centroid[index2])

        expected = [
            [0, _euclid(c, 0, 1), _euclid(c, 0, 2), _euclid(c, 0, 3)],
            [_euclid(c, 1, 0), 0, 0, _euclid(c, 1, 3)],
            [_euclid(c, 2, 0), 0, 0, _euclid(c, 2, 3)],
            [_euclid(c, 3, 0), _euclid(c, 3, 1), _euclid(c, 3, 2), 0],
        ]

        with self.test_session() as sess:
            adjacency_matrix = adjacency_euclid_distance(segmentation)
            self.assertAllEqual(adjacency_matrix.eval(), expected)
