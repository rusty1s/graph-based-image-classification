import tensorflow as tf
import numpy as np
from skimage.segmentation import slic as _slic


def slic(image, num_superpixels, compactness=1.0, max_iterations=10,
         sigma=0.0):

    def _slic_image(image):
        superpixels = _slic(image, n_segments=num_superpixels,
                            compactness=compactness, max_iter=max_iterations,
                            sigma=sigma, slic_zero=False)
        # py_func expects a float as out type.
        return superpixels.astype(np.float32)

    image = tf.cast(image, tf.int32)
    superpixels = tf.py_func(_slic_image, [image], tf.float32, stateful=False,
                             name='slic')
    return tf.cast(superpixels, tf.int32)


def slico(image, num_superpixels, compactness=1.0, max_iterations=10,
          sigma=0.0):

    def _slico_image(image):
        superpixels = _slic(image, n_segments=num_superpixels,
                            compactness=compactness, max_iter=max_iterations,
                            sigma=sigma, slic_zero=True)
        # py_func expects a float as out type
        return superpixels.astype(np.float32)

    image = tf.cast(image, tf.int32)
    superpixels = tf.py_func(_slico_image, [image], tf.float32, stateful=False,
                             name='slico')
    return tf.cast(superpixels, tf.int32)
