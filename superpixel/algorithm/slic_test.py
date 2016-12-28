import tensorflow as tf

from .slic import (slic, slico)


class SlicTest(tf.test.TestCase):

    def test_slic(self):
        image = tf.constant([
            [255, 255, 0,   0],
            [255, 255, 0,   0],
            [0,   0,   255, 255],
            [0,   0,   255, 255],
        ])

        expected = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ]

        with self.test_session() as sess:
            superpixels = slic(image, 4)
            self.assertAllEqual(superpixels.eval(), expected)

    def test_slico(self):
        image = tf.constant([
            [255, 255, 0,   0],
            [255, 255, 0,   0],
            [0,   0,   255, 255],
            [0,   0,   255, 255],
        ])

        expected = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ]

        with self.test_session() as sess:
            superpixels = slico(image, 4)
            self.assertAllEqual(superpixels.eval(), expected)
