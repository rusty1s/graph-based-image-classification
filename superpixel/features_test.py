import tensorflow as tf

from .features import features


class FeaturesTest(tf.test.TestCase):

    def test_features(self):
        image = tf.constant([
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        ])

        segmentation = tf.constant([
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 2, 2, 2],
        ], dtype=tf.int32)

        with self.test_session() as sess:
            print(features(segmentation, image).eval())
