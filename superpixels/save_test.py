from nose.tools import *

import os
import cv2

from .extract import extract_superpixels
from .save import (save_superpixel_image, save_superpixels, read_superpixels)

from .slic import image_to_slic_zero


def test_save_superpixel_test_image():
    image = cv2.imread('./superpixels/test.png')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 4))

    save_superpixel_image(image, superpixels, path='.', name='test-SLIC.png')

    assert_true(os.path.exists('./test-SLIC.png'))
    os.remove('./test-SLIC.png')


def test_save_superpixel_tree_image():
    image = cv2.imread('./superpixels/tree.jpg')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 200))

    save_superpixel_image(image, superpixels, path='./superpixels',
                          name='tree-SLIC.jpg', show_contour=True,
                          contour_color=(0, 0, 255), contour_thickness=2,
                          show_center=True, center_radius=2,
                          center_color=(0, 0, 255), show_mean=True)

    assert_true(os.path.exists('./superpixels/tree-SLIC.jpg'))

    slic_image = cv2.imread('superpixels/tree-SLIC.jpg')

    for s in superpixels.values():
        x, y = s.rounded_absolute_center
        center_color = slic_image[y][x]

        # Center should have a NEARLY red color.
        assert_in(center_color[0], range(0, 15))
        assert_in(center_color[1], range(0, 15))
        assert_in(center_color[2], range(240, 256))

    os.remove('./superpixels/tree-SLIC.jpg')
