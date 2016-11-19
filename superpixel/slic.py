""" SLIC superpixel segmentation
"""

from skimage.io import imread
from skimage.util import img_as_float
from skimage.segmentation import slic


def image_to_slic(image, segments, compactness=10.0, max_iterations=10,
                  sigma=0.0):
    return slic(image, n_segments=segments, compactness=compactness,
                max_iter=max_iterations, sigma=sigma, slic_zero=True)
