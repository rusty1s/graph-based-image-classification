import tensorflow as tf
import numpy as np
from skimage.segmentation import quickshift as skimage_quickshift


RATIO = 1.0
KERNEL_SIZE = 5
MAX_DISTANCE = 10
SIGMA = 0.0


def quickshift(image, ratio=RATIO, kernel_size=KERNEL_SIZE,
               max_distance=MAX_DISTANCE, sigma=SIGMA):
    """Segments an image using quickshift clustering in Color-(x,y) space.

    Args:
        image: The image.
        ratio: Float in range [0, 1] that balances color-space proximity
          and image-space-proximity. Higher values give more weight to
          colorspace (optional).
        kernel_size: Width of Gaussian kernel used in smoothing the sample
          density. Higher means fewer clusters (optional).
        max_distance: Cut-off point for data distances. Higher means fewer
          clusters (optional).
        sigma: Width of Gaussian kernel used in preprocessing (optional).

    Returns:
        Integer mask indicating segment labels.
    """

    image = tf.cast(image, tf.uint8)

    def _quickshift(image):
        segmentation = skimage_quickshift(image, ratio, kernel_size,
                                          max_distance, sigma=sigma)
        return segmentation.astype(np.int32)

    # TODO quickshift is stateful?
    return tf.py_func(_quickshift, [image], tf.int32, stateful=True,
                      name='quickshift')


def quickshift_generator(ratio=RATIO, kernel_size=KERNEL_SIZE,
                         max_distance=MAX_DISTANCE, sigma=SIGMA):
    """Generator to segment an image using quickshift clustering in Color-(x,y)
    space.

    Args:
        ratio: Float in range [0, 1] that balances color-space proximity
          and image-space proximity. Higher values give more weight to
          color-space proximity (optional).
        kernel_size: Width of Gaussian kernel used in smoothing the sample
          density. Higher means fewer clusters (optional).
        max_distance: Cut-off point for data distances. Higher means fewer
          clusters (optional).
        sigma: Width of Gaussian kernel used in preprocessing (optional).

    Returns:
        Segmentation algorithm that takes a single input image.
    """

    def _generator(image):
        return quickshift(image, ratio, kernel_size, max_distance, sigma)

    return _generator


def quickshift_json_generator(json):
    """Generator to segment an image using quickshift clustering in Color-(x,y)
    space based on a json object.

    Args:
        json: The json object with sensible defaults for missing values.

    Returns:
        Segmentation algorithm that takes a single input image.
    """

    return quickshift_generator(
        json['ratio'] if 'ratio' in json else RATIO,
        json['kernel_size'] if 'kernel_size' in json else KERNEL_SIZE,
        json['max_distance'] if 'max_distance' in json else MAX_DISTANCE,
        json['sigma'] if 'sigma' in json else SIGMA)
