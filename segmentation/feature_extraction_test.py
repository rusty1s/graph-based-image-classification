from math import sqrt, pi as PI

import tensorflow as tf

from .feature_extraction import feature_extraction


class FeaturesTest(tf.test.TestCase):

    def test_features(self):
        image = tf.constant([
            [[255, 0,   0],   [0,   0,   0],   [0,   0,   0]],
            [[0,   0,   0],   [0,   0,   0],   [255, 255, 255]],
            [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
        ])

        segmentation = tf.constant([
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=tf.int32)

        M = [[5, 2, 2, 2],
             [4, 1, 1, 1],
             [6, 1, 1, 1],
             [10, 1, 1, 1]]

        bbox = [2, 3]
        convex_area = 5
        perimeter = 4 + (1 + sqrt(2)) / 2

        Mw = [[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

        mean = [0.2, 0, 0]
        minimum = [0, 0, 0]
        maximum = [1, 0, 0]

        with self.test_session() as sess:
            f = feature_extraction(segmentation, image).eval()[0]
            self.assertEqual(len(f), 45)

            self.assertAllEqual(f[0:4], M[0])
            self.assertAllEqual(f[4:8], M[1])
            self.assertAllEqual(f[8:12], M[2])
            self.assertAllEqual(f[12:16], M[3])
            self.assertAllEqual(f[16:18], bbox)
            self.assertEqual(f[18], convex_area)
            self.assertEqual(round(f[19]*100), round(perimeter*100))
            self.assertAllEqual(f[20:24], Mw[0])
            self.assertAllEqual(f[24:28], Mw[1])
            self.assertAllEqual(f[28:32], Mw[2])
            self.assertAllEqual(f[32:36], Mw[3])
            self.assertAllEqual(round(f[36]*100), round(mean[0]*100))
            self.assertAllEqual(f[37:39], mean[1:3])
            self.assertAllEqual(f[39:42], minimum)
            self.assertAllEqual(f[42:45], maximum)
