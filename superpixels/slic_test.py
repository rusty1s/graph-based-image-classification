from nose.tools import *
from numpy import testing as np_test

import cv2
import numpy as np

from .slic import (image_to_slic, image_to_slic_zero)
from .extract import extract_superpixels


def test_image_to_slic():
    image = cv2.imread('superpixels/test.png')
    assert_is_not_none(image)

    slic = image_to_slic(image, 4)

    assert_equals(slic.shape, (20, 20))
    assert_equals(len(extract_superpixels(image, slic)), 4)


def test_image_to_slic_zero():
    image = cv2.imread('superpixels/test.png')
    assert_is_not_none(image)

    slic = image_to_slic_zero(image, 4)

    assert_equals(slic.shape, (20, 20))
    assert_equals(len(extract_superpixels(image, slic)), 4)
