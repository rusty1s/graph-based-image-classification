import tensorflow as tf
import numpy as np
from skimage.segmentation import slic as skimage_slic


NUM_SUPERPIXELS = 400
COMPACTNESS = 30.0
MAX_ITERATIONS = 10
SIGMA = 0.0
MIN_SIZE_FACTOR = 0.5
MAX_SIZE_FACTOR = 3.0
ENFORCE_CONNECTIVITY = True


def slico(image, num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
          max_iterations=MAX_ITERATIONS, sigma=SIGMA,
          min_size_factor=MIN_SIZE_FACTOR, max_size_factor=MAX_SIZE_FACTOR,
          enforce_connectivity=ENFORCE_CONNECTIVITY):

    image = tf.cast(image, tf.uint8)

    def _slico(image):
        segmentation = skimage_slic(image, num_superpixels, compactness,
                                    max_iterations, sigma,
                                    min_size_factor=min_size_factor,
                                    max_size_factor=max_size_factor,
                                    enforce_connectivity=enforce_connectivity,
                                    slic_zero=True)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(
        _slico, [image], tf.float32, stateful=False, name='slico')

    return tf.cast(segmentation, tf.int32)


def slico_generator(num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS,
                    max_iterations=MAX_ITERATIONS, sigma=SIGMA,
                    min_size_factor=MIN_SIZE_FACTOR,
                    max_size_factor=MAX_SIZE_FACTOR,
                    enforce_connectivity=ENFORCE_CONNECTIVITY):

    def _generator(image):
        return slico(image, num_superpixels, compactness, max_iterations,
                     sigma)

    return _generator


def slico_json_generator(json):
    return slico_generator(
        json['num_superpixels'] if 'num_superpixels' in json
        else NUM_SUPERPIXELS,
        json['compactness'] if 'compactness' in json else COMPACTNESS,
        json['max_iterations'] if 'max_iterations' in json else MAX_ITERATIONS,
        json['sigma'] if 'sigma' in json else SIGMA,
        json['min_size_factor'] if 'min_size_factor' in json
        else MIN_SIZE_FACTOR,
        json['max_size_factor'] if 'max_size_factor' in json
        else MAX_SIZE_FACTOR,
        json['enforce_connectivity'] if 'enforce_connectivity' in json
        else ENFORCE_CONNECTIVITY)
