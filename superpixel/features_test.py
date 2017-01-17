import tensorflow as tf

from .features import segmentation_features


class FeaturesTest(tf.test.TestCase):

    def test_features(self):
        segmentation = tf.constant([
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
        ], dtype=tf.int32)

        with self.test_session() as sess:
            print(segmentation_features(segmentation).eval())
