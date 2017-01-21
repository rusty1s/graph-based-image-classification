import tensorflow as tf
import numpy as np
from skimage.segmentation import slic as skimage_slic


NUM = 400
COMPACTNESS = 30.0
MAX_ITER = 10
SIGMA = 0.0


def slico(image, num_superpixels=None, compactness=None, max_iterations=None,
          sigma=None):

    num_superpixels = NUM if num_superpixels is None else num_superpixels
    compactness = COMPACTNESS if compactness is None else compactness
    max_iterations = MAX_ITER if max_iterations is None else max_iterations
    sigma = SIGMA if sigma is None else sigma

    image = tf.cast(image, tf.uint8)

    def _slico(image):
        segmentation = skimage_slic(image, num_superpixels, compactness,
                                    max_iterations, sigma, slic_zero=True)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(
        _slico, [image], tf.float32, stateful=False, name='slico')

    return tf.cast(segmentation, tf.int32)


def slico_generator(num_superpixels=None, compactness=None,
                    max_iterations=None, sigma=None):

    def _generator(image):
        return slico(image, num_superpixels, compactness, max_iterations,
                     sigma)

    return _generator


def slico_json_generator(json):
    return slico_generator(
        json['num_superpixels'] if 'num_superpixels' in json else None,
        json['compactness'] if 'compactness' in json else None,
        json['max_iterations'] if 'max_iterations' in json else None,
        json['sigma'] if 'sigma' in json else None)
