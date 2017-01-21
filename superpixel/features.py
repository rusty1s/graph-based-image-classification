import tensorflow as tf
import numpy as np
from skimage.measure import regionprops


NUM_FEATURES = 5


def features(segmentation, image):
    def _features(segmentation, intensity_image):
        props = regionprops(segmentation, intensity_image)
        f = np.zeros((len(props), NUM_FEATURES), dtype=np.float32)

        for i, prop in enumerate(props):
            pass

        return f

    # We need to increment the segmentation, because labels with value 0 are
    # ignored when calling regionprops.
    segmentation = segmentation + tf.ones_like(segmentation)

    # Get the intensity image with shape [height, width] of the image.
    with tf.name_scope('intensity_image', values=[image]):
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.rgb_to_hsv(image)
        image = tf.strided_slice(image, [0, 0, 2],
                                 [tf.shape(image)[0], tf.shape(image)[1], 3],
                                 [1, 1, 1])
        image = tf.squeeze(image)

    return tf.py_func(_features, [segmentation, image], tf.float32,
                      stateful=False, name='segmentation_features')
