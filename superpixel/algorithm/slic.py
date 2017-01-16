import tensorflow as tf
import numpy as np
from skimage.segmentation import slic as _slic

COMPACTNESS = 1.0
MAX_ITERATIONS = 10
SIGMA = 0.0


def _slic(image, slic_zero, num_superpixels, compactness=COMPACTNESS,
          max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    def _slic_image(image):
        superpixels = _slic(image, n_segments=num_superpixels,
                            compactness=compactness, max_iter=max_iterations,
                            sigma=sigma, slic_zero=slic_zero)
        # py_func expects a float as out type.
        return superpixels.astype(np.float32)

    image = tf.cast(image, tf.uint8)
    superpixels = tf.py_func(_slic_image, [image], tf.float32, stateful=False,
                             name='slico' if slic_zero else 'slic')
    return tf.cast(superpixels, tf.int32)


def slic(image, num_superpixels, compactness=COMPACTNESS,
         max_iterations=MAX_ITERATIONS, sigma=SIGMA):
    return _slic(
        image, True, num_superpixels, compactness, max_iterations, sigma)


def slico(image, num_superpixels, compactness=COMPACTNESS,
          max_iterations=MAX_ITERATIONS, sigma=SIGMA):
    return _slic(
        image, False, num_superpixels, compactness, max_iterations, sigma)


def slic_generator(num_superpixels, compactness=COMPACTNESS,
                   max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    def _generator(image):
        return slic(image, num_superpixels, compactness, max_iterations, sigma)

    return _generator


def slico_generator(num_superpixels, compactness=COMPACTNESS,
                    max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    def _generator(image):
        return slico(image, num_superpixels, compactness, max_iterations,
                     sigma)

    return _generator
