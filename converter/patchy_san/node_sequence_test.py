import tensorflow as tf

from .node_sequence import node_sequence


class NodeSequenceTest(tf.test.TestCase):

    def test_node_sequence(self):
        sequence = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])

        with self.test_session() as sess:
            s = node_sequence(sequence, width=4, stride=1)
            self.assertAllEqual(s.eval(), [1, 2, 3, 4])

            s = node_sequence(sequence, width=10, stride=1)
            self.assertAllEqual(s.eval(), [1, 2, 3, 4, 5, 6, 7, 8, -1, -1])

            s = node_sequence(sequence, width=4, stride=2)
            self.assertAllEqual(s.eval(), [1, 3, 5, 7])

            s = node_sequence(sequence, width=5, stride=2)
            self.assertAllEqual(s.eval(), [1, 3, 5, 7, -1])

            s = node_sequence(sequence, width=5, stride=3)
            self.assertAllEqual(s.eval(), [1, 4, 7, -1, -1])
