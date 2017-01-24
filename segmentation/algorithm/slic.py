import tensorflow as tf
import numpy as np
from skimage.segmentation import slic as skimage_slic


NUM_SEGMENTS = 400
COMPACTNESS = 30.0
MAX_ITERATIONS = 10
SIGMA = 0.0
MIN_SIZE_FACTOR = 0.5
MAX_SIZE_FACTOR = 3.0
ENFORCE_CONNECTIVITY = True


def slic(image, num_segments=NUM_SEGMENTS, compactness=COMPACTNESS,
         max_iterations=MAX_ITERATIONS, sigma=SIGMA,
         min_size_factor=MIN_SIZE_FACTOR, max_size_factor=MAX_SIZE_FACTOR,
         enforce_connectivity=ENFORCE_CONNECTIVITY):
    """Segments an image using k-means clustering in Color-(x,y,z) space.

    Args:
        image: The image.
        num_segments: The (approiximate) number of segments in the segmented
          output image (optional).
        compactness: Balances color-space proximity and image-space-proximity.
          Higher values give more weight to image-space proximity (optional).
        max_iterations: Maximum number of iterations of k-means.
        sigma: Width of Gaussian kernel used in preprocessing (optional).
        min_size_factor: Proportion of the minimum segment size to be removed
          with respect to the supposed segment size
          `depth*width*height/num_segments` (optional).
        max_size_factor: Proportion of the maximum connected segment size
          (optional).
        enforce_connectivitiy: Whether the generated segments are connected or
          not (optional).

    Returns:
        Integer mask indicating segment labels.
    """

    image = tf.cast(image, tf.uint8)

    def _slic(image):
        segmentation = skimage_slic(image, num_segments, compactness,
                                    max_iterations, sigma,
                                    min_size_factor=min_size_factor,
                                    max_size_factor=max_size_factor,
                                    enforce_connectivity=enforce_connectivity,
                                    slic_zero=False)
        return segmentation.astype(np.int32)

    return tf.py_func(_slic, [image], tf.int32, stateful=False, name='slic')


def slic_generator(num_segments=NUM_SEGMENTS, compactness=COMPACTNESS,
                   max_iterations=MAX_ITERATIONS, sigma=SIGMA,
                   min_size_factor=MIN_SIZE_FACTOR,
                   max_size_factor=MAX_SIZE_FACTOR,
                   enforce_connectivity=ENFORCE_CONNECTIVITY):
    """Generator to segment an image using k-means clustering in Color-(x,y,z)
    space.

    Args:
        num_segments: The (approiximate) number of segments in the segmented
          output image (optional).
        compactness: Balances color-space proximity and image-space-proximity.
          Higher values give more weight to image-space proximity (optional).
        max_iterations: Maximum number of iterations of k-means.
        sigma: Width of Gaussian kernel used in preprocessing (optional).
        min_size_factor: Proportion of the minimum segment size to be removed
          with respect to the supposed segment size
          `depth*width*height/num_segments` (optional).
        max_size_factor: Proportion of the maximum connected segment size
          (optional).
        enforce_connectivitiy: Whether the generated segments are connected or
          not (optional).

    Returns:
        Segmentation algorithm that takes a single image as argument.
    """

    def _generator(image):
        return slic(image, num_segments, compactness, max_iterations, sigma)

    return _generator


def slic_json_generator(json):
    """Generator to segment an image using k-means clustering in Color-(x,y,z)
    space based on a json object.

    Args:
        json: The json object with sensible defaults for missing values.

    Returns:
        Segmentation algorithm that takes a single image as argument.
    """

    return slic_generator(
        json['num_segments'] if 'num_segments' in json
        else NUM_SEGMENTS,
        json['compactness'] if 'compactness' in json else COMPACTNESS,
        json['max_iterations'] if 'max_iterations' in json else MAX_ITERATIONS,
        json['sigma'] if 'sigma' in json else SIGMA,
        json['min_size_factor'] if 'min_size_factor' in json
        else MIN_SIZE_FACTOR,
        json['max_size_factor'] if 'max_size_factor' in json
        else MAX_SIZE_FACTOR,
        json['enforce_connectivity'] if 'enforce_connectivity' in json
        else ENFORCE_CONNECTIVITY)
