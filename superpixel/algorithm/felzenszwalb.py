import tensorflow as tf
import numpy as np
from skimage.segmentation import felzenszwalb as skimage_felzenszwalb


SCALE = 1.0
SIGMA = 0.8
MIN_SIZE = 20


def felzenszwalb(image, scale=None, sigma=None, min_size=None):

    scale = SCALE if scale is None else scale
    sigma = SIGMA if sigma is None else sigma
    min_size = MIN_SIZE if min_size is None else min_size

    image = tf.cast(image, tf.uint8)

    def _felzenszwalb(image):
        segmentation = skimage_felzenszwalb(image, scale, sigma, min_size)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(_felzenszwalb, [image], tf.float32,
                              stateful=False, name='felzenszwalb')

    return tf.cast(segmentation, tf.int32)


def felzenszwalb_generator(scale=None, sigma=None, min_size=None):

    def _generator(image):
        return felzenszwalb(image, scale, sigma, min_size)

    return _generator


def felzenszwalb_json_generator(json):
    return felzenszwalb_generator(
        json['scale'] if 'scale' in json else None,
        json['sigma'] if 'sigma' in json else None,
        json['min_size'] if 'min_size' in json else None)
