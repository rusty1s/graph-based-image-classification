import tensorflow as tf
import numpy as np
from skimage.segmentation import felzenszwalb as skimage_felzenszwalb


SCALE = 1.0
SIGMA = 0.0
MIN_SIZE = 20


def felzenszwalb(image, scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE):
    """Computes Felsenszwalb's efficient graph based image segmentation.

    Args:
        image: The image.
        scale: Float indicating largeness of clusters (optional).
        sigma: Width of Gaussian kernel used in preprocessing (optional).
        min_size: Minimum component size. Enforced using postprocessing
          (optional).

    Returns:
        Integer mask indicating segment labels.
    """

    image = tf.cast(image, tf.uint8)

    def _felzenszwalb(image):
        segmentation = skimage_felzenszwalb(image, scale, sigma, min_size)

        # py_func expects a float as out type.
        return segmentation.astype(np.float32)

    segmentation = tf.py_func(_felzenszwalb, [image], tf.float32,
                              stateful=False, name='felzenszwalb')

    return tf.cast(segmentation, tf.int32)


def felzenszwalb_generator(scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE):
    """Generator to compute Felsenszwalb's efficient graph based image
    segmentation.

    Args:
        scale: Float indicating largeness of clusters (optional).
        sigma: Width of Gaussian kernel used in preprocessing (optional).
        min_size: Minimum component size. Enforced using postprocessing
          (optional).

    Returns:
        Segmentation algorithm that takes a single image as argument.
    """

    def _generator(image):
        return felzenszwalb(image, scale, sigma, min_size)

    return _generator


def felzenszwalb_json_generator(json):
    """Generator to compute Felsenszwalb's efficient graph based image
    segmentation based on a json object.

    Args:
        json: The json object with sensible defaults for missing values.

    Returns:
        Segmentation algorithm that takes a single image as argument.
    """

    return felzenszwalb_generator(
        json['scale'] if 'scale' in json else SCALE,
        json['sigma'] if 'sigma' in json else SIGMA,
        json['min_size'] if 'min_size' in json else MIN_SIZE)
