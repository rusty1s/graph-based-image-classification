from math import sqrt, pi as PI

import tensorflow as tf
import numpy as np
from skimage.measure import regionprops


# Static number of features to generate for one segment.
NUM_FEATURES = 83


# TODO Oriented Bounding Box missing
def feature_extraction(segmentation, image):
    def _feature_extraction(segmentation, intensity_image, image):
        props = regionprops(segmentation, intensity_image)

        # Create the output feature vector with shape [num_segments,
        # num_features].
        features = np.zeros((len(props), NUM_FEATURES), dtype=np.float32)

        for i, prop in enumerate(props):
            area = prop['area']
            features[i][0] = area

            bbox = prop['bbox']
            features[i][1] = bbox[2] - bbox[0]  # bbox height
            features[i][2] = bbox[3] - bbox[1]  # bbox height

            features[i][3] = prop['convex_area']
            features[i][4] = prop['eccentricity']
            features[i][5] = prop['equivalent_diameter']
            features[i][6] = prop['euler_number']
            features[i][7] = prop['extent']
            features[i][8] = prop['filled_area']

            inertia_tensor = prop['inertia_tensor']
            features[i][9] = inertia_tensor[0][0]  # mu'_20
            features[i][10] = inertia_tensor[0][1]  # mu'_11
            features[i][11] = inertia_tensor[1][1]  # mu'_02

            inertia_tensor_eigvals = prop['inertia_tensor_eigvals']
            features[i][12] = inertia_tensor_eigvals[0]  # lambda_1
            features[i][13] = inertia_tensor_eigvals[1]  # lambda_2

            local_centroid = prop['local_centroid']
            features[i][14] = local_centroid[0]
            features[i][15] = local_centroid[1]

            features[i][16] = prop['major_axis_length']
            features[i][17] = prop['minor_axis_length']

            features[i][18] = prop['max_intensity']
            features[i][19] = prop['mean_intensity']
            features[i][20] = prop['min_intensity']

            mu = prop['moments_central']
            features[i][21] = mu[0][0]
            features[i][22] = mu[1][1]
            features[i][23] = mu[2][0]
            features[i][24] = mu[0][2]
            features[i][25] = mu[2][1]
            features[i][26] = mu[1][2]
            features[i][27] = mu[2][2]

            eta = prop['moments_normalized']
            features[i][28] = eta[0][0]
            features[i][29] = eta[1][1]
            features[i][30] = eta[2][0]
            features[i][31] = eta[0][2]
            features[i][32] = eta[2][1]
            features[i][33] = eta[1][2]
            features[i][34] = eta[2][2]

            hu = prop['moments_hu']
            features[i][35] = hu[0]
            features[i][36] = hu[1]
            features[i][37] = hu[2]
            features[i][38] = hu[3]
            features[i][39] = hu[4]
            features[i][40] = hu[5]
            features[i][41] = hu[6]

            perimeter = prop['perimeter']
            features[i][42] = perimeter

            features[i][43] = prop['orientation']
            features[i][44] = prop['solidity']

            weighted_local_centroid = prop['weighted_local_centroid']
            features[i][45] = weighted_local_centroid[0]
            features[i][46] = weighted_local_centroid[1]

            weighted_mu = prop['weighted_moments_central']
            features[i][47] = weighted_mu[0][0]
            features[i][48] = weighted_mu[1][1]
            features[i][49] = weighted_mu[2][0]
            features[i][50] = weighted_mu[0][2]
            features[i][51] = weighted_mu[2][1]
            features[i][52] = weighted_mu[1][2]
            features[i][53] = weighted_mu[2][2]

            weighted_eta = prop['weighted_moments_normalized']
            features[i][54] = weighted_eta[0][0]
            features[i][55] = weighted_eta[1][1]
            features[i][56] = weighted_eta[2][0]
            features[i][57] = weighted_eta[0][2]
            features[i][58] = weighted_eta[2][1]
            features[i][59] = weighted_eta[1][2]
            features[i][60] = weighted_eta[2][2]

            weighted_hu = prop['moments_hu']
            features[i][61] = weighted_hu[0]
            features[i][62] = weighted_hu[1]
            features[i][63] = weighted_hu[2]
            features[i][64] = weighted_hu[3]
            features[i][65] = weighted_hu[4]
            features[i][66] = weighted_hu[5]
            features[i][67] = weighted_hu[6]

            circularity = (4 * PI * area) / (perimeter**2)
            features[i][68] = circularity

            central_moment_feature = (mu[2][0] + mu[0][2]) / mu[0][0]
            features[i][69] = central_moment_feature

            elongation = (sqrt((mu[2][0] - mu[0][2])**2 + mu[1][1]**2)) /\
                         (mu[2][0] + mu[0][2])
            features[i][70] = elongation

            sliced_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            sliced_image = sliced_image[prop['image']]

            features[i][71] = sliced_image[..., 0].mean()
            features[i][72] = sliced_image[..., 1].mean()
            features[i][73] = sliced_image[..., 2].mean()

            features[i][74] = sliced_image[..., 0].sum()
            features[i][75] = sliced_image[..., 1].sum()
            features[i][76] = sliced_image[..., 2].sum()

            features[i][77] = sliced_image[..., 0].min()
            features[i][78] = sliced_image[..., 1].min()
            features[i][79] = sliced_image[..., 2].min()

            features[i][80] = sliced_image[..., 0].max()
            features[i][81] = sliced_image[..., 1].max()
            features[i][82] = sliced_image[..., 2].max()

        return features

    # We need to increment the segmentation, because labels with value 0 are
    # ignored when calling regionprops.
    segmentation = segmentation + tf.ones_like(segmentation)

    # Convert to HSV Colorspace.
    with tf.name_scope('image_to_float', values=[image]):
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Get the intensity image with shape [height, width] of the image.
    with tf.name_scope('intensity_image', values=[image]):
        intensity_image = tf.image.rgb_to_hsv(image)
        intensity_image = tf.strided_slice(
            intensitiy_image,
            [0, 0, 2],
            [tf.shape(image)[0], tf.shape(image)[1], 3],
            [1, 1, 1])
        intensity_image = tf.squeeze(intensity_image)

    return tf.py_func(
        _feature_extraction, [segmentation, intensity_image, image],
        tf.float32, stateful=False, name='feature_extraction')
