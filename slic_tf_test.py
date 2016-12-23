import tensorflow as tf

from superpixels import (slic, slico)


class SlicTest(tf.test.TestCase):

    def test_slic(self):
        # TODO: Slic doesn't seem to work well on low count segments! we need
        # at least a 10x10 matrix for 4 segments!
        image = tf.constant([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ])

        expected = ([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ])

        with self.test_session() as sess:
            superpixels = slic(image, 4)
            self.assertAllEqual(superpixels.eval(), expected)

    def test_slico(self):
        # TODO: Slico doesn't seem to work well on low count segments! we need
        # at least a 10x10 matrix for 4 segments!
        image = tf.constant([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ])

        expected = ([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ])

        with self.test_session() as sess:
            superpixels = slico(image, 4)
            self.assertAllEqual(superpixels.eval(), expected)


if __name__ == '__main__':
    tf.test.main()
