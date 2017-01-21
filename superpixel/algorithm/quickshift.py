import tensorflow as tf
import numpy as np
from skimage.segmentation import quickshift as skimage_quickshift


RATIO = 1.0
KERNEL_SIZE = 5
MAX_DISTANCE = 10
SIGMA = 0


def quickshift(image, ratio=None, kernel_size=None, max_distance=None,
               sigma=None):

    ratio = RATIO if ratio is None else ratio
    kernel_size = KERNEL_SIZE if kernel_size is None else kernel_size
    max_distance = MAX_DISTANCE if max_distance is None else max_distance
    sigma = SIGMA if sigma is None else sigma

    image = tf.cast(image, tf.uint8)

    def _quickshift(image):
        segmentation = skimage_quickshift(
            image, ratio, kernel_size, max_distance, sigma=sigma)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(
        _quickshift, [image], tf.float32, stateful=True, name='quickshift')

    return tf.cast(segmentation, tf.int32)


def quickshift_generator(ratio=None, kernel_size=None, max_distance=None,
                         sigma=None):

    def _generator(image):
        return quickshift(image, ratio, kernel_size, max_distance, sigma)

    return _generator


def quickshift_json_generator(json):
    return quickshift_generator(
        json['ratio'] if 'ratio' in json else None,
        json['kernel_size'] if 'kernel_size' in json else None,
        json['max_distance'] if 'max_distance' in json else None,
        json['sigma'] if 'sigma' in json else None)
