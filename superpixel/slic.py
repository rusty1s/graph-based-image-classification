""" SLIC superpixel segmentation
"""

from skimage.segmentation import slic


def image_to_slic(image, segments, compactness=10.0, max_iterations=10,
                  sigma=0.0):
    """Segments image using k-means clustering in Color-(x,y,z) space."""

    return slic(image, n_segments=segments, compactness=compactness,
                max_iter=max_iterations, sigma=sigma, slic_zero=True)
