"""Methods for working with superpixels
"""

import imutils
import cv2
import numpy as np


def get_segment_values(segments):
    return np.unique(segments)


def get_mask(segments, segment_value):
    mask = np.zeros(segments, dtype=np.uint8)
    mask[segments == segment_value] = 255
    return mask


def get_masks(segments):
    segment_values = get_segment_values(segments)

    return list(map(lambda x: get_mask(segments, x), segment_values))


def get_centroid(mask):
    pass


def get_centroids(masks):
    pass


def get_mean_color(image, mask):
    return cv2.mean(image, mask)
