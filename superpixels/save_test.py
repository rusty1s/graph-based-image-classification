from nose.tools import *

import os
import cv2

from .extract import extract_superpixels
from .save import (save_superpixel_image, save_superpixels, load_superpixels)

from .slic import image_to_slic_zero


def test_save_superpixel_test_image():
    image = cv2.imread('./superpixels/test.png')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 4))

    save_superpixel_image(image, superpixels, path='.', name='test-SLIC.png',
                          show_center=True)

    assert_true(os.path.exists('./test-SLIC.png'))

    slic_image = cv2.imread('./test-SLIC.png')
    assert_is_not_none(slic_image)

    for s in superpixels:
        x, y = s.rounded_absolute_center
        center_color = slic_image[y][x]

        # Center should have a default black color.
        assert_equals(center_color[0], 0)
        assert_equals(center_color[1], 0)
        assert_equals(center_color[2], 255)

    os.remove('./test-SLIC.png')


def test_save_superpixel_tree_image():
    image = cv2.imread('./superpixels/tree.jpg')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 200))

    save_superpixel_image(image, superpixels, path='./superpixels',
                          name='tree-SLIC.jpg', show_contour=True,
                          contour_color=(0, 0, 0), contour_thickness=2,
                          show_center=True, center_radius=2,
                          center_color=(0, 0, 0), show_mean=True)

    assert_true(os.path.exists('./superpixels/tree-SLIC.jpg'))

    slic_image = cv2.imread('superpixels/tree-SLIC.jpg')
    assert_is_not_none(slic_image)

    for s in superpixels:
        x, y = s.rounded_absolute_center
        center_color = slic_image[y][x]

        # Center should have a NEARLY red color, because of jpeg compression.
        assert_in(center_color[0], range(0, 25))
        assert_in(center_color[1], range(0, 25))
        assert_in(center_color[2], range(0, 25))

    os.remove('./superpixels/tree-SLIC.jpg')


def test_save_superpixels():
    image = cv2.imread('./superpixels/test.png')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 4))

    save_superpixels(superpixels, '.', 'test.pkl')

    assert_true(os.path.exists('./test.pkl'))

    os.remove('./test.pkl')


def test_load_superpixels():
    image = cv2.imread('./superpixels/test.png')
    assert_is_not_none(image)

    superpixels = extract_superpixels(image, image_to_slic_zero(image, 4))
    assert_equals(len(superpixels), 4)

    save_superpixels(superpixels, '.', 'test.pkl')

    assert_true(os.path.exists('./test.pkl'))

    loaded_superpixels = load_superpixels('./test.pkl')
    assert_equals(len(loaded_superpixels), 4)

    os.remove('./test.pkl')
