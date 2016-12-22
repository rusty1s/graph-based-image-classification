import tensorflow as tf

from superpixels import slic


class SlicTest(tf.test.TestCase):

    def test_slic(self):
        image = tf.constant([1, 2, 3, 4, 5, 6])

        with self.test_session() as sess:
            superpixel = slic(image)
            print(superpixel.eval())


if __name__ == '__main__':
    tf.test.main()
