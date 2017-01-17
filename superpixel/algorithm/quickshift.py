import tensorflow as tf
import numpy as np
from skimage.segmentation import quickshift as skimage_quickshift


RATIO = 1.0
KERNEL_SIZE = 5
MAX_DISTANCE = 10
SIGMA = 0


def quickshift(image, ratio=RATIO, kernel_size=KERNEL_SIZE,
               max_distance=MAX_DISTANCE, sigma=SIGMA):

    image = tf.cast(image, tf.uint8)

    def _quickshift(image):
        segmentation = skimage_quickshift(image, ratio, kernel_size,
                                          max_distance, sigma=sigma)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(_quickshift, [image], tf.float32, stateful=False,
                              name='quickshift')

    return tf.cast(segmentation, tf.int32)


def quickshift_generator(ratio=RATIO, kernel_size=KERNEL_SIZE,
                         max_distance=MAX_DISTANCE, sigma=SIGMA):

    def _generator(image):
        return quickshift(image, ratio, kernel_size, max_distance, sigma)

    return _generator
