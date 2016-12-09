from nose.tools import *

import os
import cv2

from .extract import extract_superpixels
from .save import (save_superpixel_image, save_superpixels, read_superpixels)

from .slic import image_to_slic_zero


def test_save_superpixel_image_1():
    image = cv2.imread('superpixels/tree.jpg')
    assert_is_not_none(image)

    save_superpixel_image(image, image_to_slic_zero(image, 200), '.',
                          'tree-SLIC.jpg')

    assert_true(os.path.exists('./tree-SLIC.jpg'))
    os.remove('./tree-SLIC.jpg')


def test_save_superpixel_image_2():
    image = cv2.imread('superpixels/test.png')
    assert_is_not_none(image)

    save_superpixel_image(image, image_to_slic_zero(image, 4), '.',
                          'test-SLIC.png')

    assert_true(os.path.exists('./test-SLIC.png'))
    os.remove('./test-SLIC.png')
