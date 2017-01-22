import tensorflow as tf

from .felzenszwalb import felzenszwalb


class FelzenszwalbTest(tf.test.TestCase):

    def test_felzenszwalb(self):
        image = tf.constant([
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        ])

        expected = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        with self.test_session() as sess:
            segmentation = felzenszwalb(image)
            self.assertAllEqual(segmentation.eval(), expected)
