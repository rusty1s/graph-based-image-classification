""" SLIC superpixel segmentation"""

from skimage.segmentation import slic

# **SLIC-algorithm:**
#
# Segments image using k-means clustering in Color-(x,y,z) space.
#
# - image: Input image, which can be 2D or 3D, and grayscale or multichannel.
# - num_superpixels: The (approximate) number of superpixels in the output
#   image.
# - compactness: Balances color proximity and space proximity. A higher value
#   value gives more weight to space proximity making superpixel shapes more
#   square/cubic. Default: 1.0.
# - max_iterations: Maximum number of iterations of k-means. Default: 10.
# - sigma: Width of gaussian smoothing kernel for pre-processing. Default: 0.0.


def image_to_slic(image, num_superpixels, compactness=1.0, max_iterations=10,
                  sigma=0.0):
    """Segments image using the SLIC algorithm."""

    return slic(image, n_segments=num_superpixels, compactness=compactness,
                max_iter=max_iterations, sigma=sigma, slic_zero=False)


def image_to_slic_zero(image, num_superpixels, compactness=1.0,
                       max_iterations=10, sigma=0.0):
    """Segments image using the SLIC-zero algorithm, that is the zero-parameter
    mode of SLIC."""

    return slic(image, n_segments=num_superpixels, compactness=compactness,
                max_iter=max_iterations, sigma=sigma, slic_zero=True)
