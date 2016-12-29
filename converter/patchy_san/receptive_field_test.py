import tensorflow as tf

from .receptive_field import (receptive_field, receptive_fields)


class ReceptiveFieldTest(tf.test.TestCase):

    def test_receptive_field(self):
        neighborhood = tf.constant([-1, 0, 2, 1])

        nodes = tf.constant([
            [1.0, 0.5],
            [3.0, 0.0],
            [1.5, 1.0],
        ])

        expected = [[0.0, 0.0], [1.0, 0.5], [1.5, 1.0], [3.0, 0.0]]

        with self.test_session() as sess:
            field = receptive_field(neighborhood, nodes, 2)
            self.assertAllEqual(field.eval(), expected)

    def test_receptive_fields(self):
        neighborhoods = tf.constant([
            [-1, 0, 2, 1],
            [1, 0, -1, 2],
        ])

        nodes = tf.constant([
            [1.0, 0.5],
            [3.0, 0.0],
            [1.5, 1.0],
        ])

        expected = [
            [[0.0, 0.0], [1.0, 0.5], [1.5, 1.0], [3.0, 0.0]],
            [[3.0, 0.0], [1.0, 0.5], [0.0, 0.0], [1.5, 1.0]],
        ]

        with self.test_session() as sess:
            fields = receptive_fields(neighborhoods, nodes, 2)
            self.assertAllEqual(fields.eval(), expected)
