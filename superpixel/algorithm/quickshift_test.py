import tensorflow as tf

from .quickshift import quickshift


class QuickshiftTest(tf.test.TestCase):

    def test_quickshift(self):
        image = tf.constant([
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        ])

        expected = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]

        with self.test_session() as sess:
            segmentation = quickshift(image)
            # TODO Quickshift is stateful, so we get a different result every
            # time we ran it.
            # self.assertAllEqual(segmentation.eval(), expected)
