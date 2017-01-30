from math import sqrt, pi as PI, isnan

import tensorflow as tf
import numpy as np
from skimage.measure import regionprops


# Static number of features to generate for one segment.
NUM_FEATURES = 45


# TODO Oriented Bounding Box missing
def feature_extraction(segmentation, image):
    """Extracts a fixed number of features for every segment label in the
    segmentation.

    Args:
        segmentation: The segmentation.
        image: The corresponding original image.

    Returns:
        Numpy array with shape [num_segments, num_features].
    """

    def _feature_extraction(segmentation, intensity_image, image):
        props = regionprops(segmentation, intensity_image)

        # Create the output feature vector with shape
        # [num_segments, num_features].
        features = np.zeros((len(props), NUM_FEATURES), dtype=np.float32)

        for i, prop in enumerate(props):
            # Moments features.
            features[i][0:16] = prop['moments'].flatten()

            # Bounding box features.
            bbox = prop['bbox']
            features[i][16] = bbox[2] - bbox[0]
            features[i][17] = bbox[3] - bbox[1]

            # Polygon features.
            features[i][18] = prop['convex_area']
            features[i][19] = prop['perimeter']

            # Weighted moments features.
            features[i][20:36] = prop['weighted_moments'].flatten()

            # Color features.
            sliced_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            sliced_image = sliced_image[prop['image']]

            features[i][36] = sliced_image[..., 0].mean()
            features[i][37] = sliced_image[..., 1].mean()
            features[i][38] = sliced_image[..., 2].mean()

            features[i][39] = sliced_image[..., 0].min()
            features[i][40] = sliced_image[..., 1].min()
            features[i][41] = sliced_image[..., 2].min()

            features[i][42] = sliced_image[..., 0].max()
            features[i][43] = sliced_image[..., 1].max()
            features[i][44] = sliced_image[..., 2].max()

        return features

    # We need to increment the segmentation, because labels with value 0 are
    # ignored when calling regionprops.
    segmentation = segmentation + tf.ones_like(segmentation)

    # Convert to uint8 image to float representation in the range [0, 1].
    with tf.name_scope('image_to_float', values=[image]):
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Get the intensity image with shape [height, width] of the image by
    # converting it to the HSV Colorspace.
    with tf.name_scope('intensity_image', values=[image]):
        intensity_image = tf.image.rgb_to_hsv(image)
        intensity_image = tf.strided_slice(
            intensity_image,
            [0, 0, 2],
            [tf.shape(image)[0], tf.shape(image)[1], 3],
            [1, 1, 1])
        intensity_image = tf.squeeze(intensity_image)

    return tf.py_func(
        _feature_extraction, [segmentation, intensity_image, image],
        tf.float32, stateful=False, name='feature_extraction')
