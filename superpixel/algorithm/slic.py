import tensorflow as tf
import numpy as np
from skimage.segmentation import slic as skimage_slic


NUM_SUPERPIXELS = 400
COMPACTNESS = 30.0
MAX_ITERATIONS = 10
SIGMA = 0.0


def _slic(image, num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
          max_iterations=MAX_ITERATIONS, sigma=SIGMA, slic_zero=False):

    image = tf.cast(image, tf.uint8)

    def _slic_image(image):
        segmentation = skimage_slic(image, num_superpixels, compactness,
                                    max_iterations, sigma, slic_zero=slic_zero)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(_slic_image, [image], tf.float32, stateful=False,
                              name='slico' if slic_zero else 'slic')

    return tf.cast(segmentation, tf.int32)


def slic(image, num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
         max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    return _slic(image, num_superpixels, compactness, max_iterations, sigma,
                 slic_zero=False)


def slico(image, num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
          max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    return _slic(image, num_superpixels, compactness, max_iterations, sigma,
                 slic_zero=True)


def slic_generator(num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
                   max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    def _generator(image):
        return slic(image, num_superpixels, compactness, max_iterations, sigma)

    return _generator


def slico_generator(num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
                    max_iterations=MAX_ITERATIONS, sigma=SIGMA):

    def _generator(image):
        return slico(image, num_superpixels, compactness, max_iterations,
                     sigma)

    return _generator
