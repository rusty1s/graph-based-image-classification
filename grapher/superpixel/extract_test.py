from nose.tools import *
from numpy import testing as np_test

import numpy as np

from .extract import extract

image = np.array([
        [0,   10,  20,  30],
        [40,  50,  60,  70],
        [80,  90,  100, 110],
        [120, 130, 140, 150],
    ])

superpixel_representation = np.array([
        [1, 1, 2, 1],
        [1, 1, 1, 1],
        [3, 3, 3, 1],
        [3, 4, 4, 4],
    ])


def test_extract():
    superpixels = extract(image, superpixel_representation)

    assert_equals(len(superpixels), 4)


def test_superpixel_1():
    superpixel = extract(image, superpixel_representation)[0]

    assert_equals(superpixel.id, 1)
    assert_equals(superpixel.order, 0)

    assert_equals(superpixel.left, 0)
    assert_equals(superpixel.top, 0)
    assert_equals(superpixel.width, 4)
    assert_equals(superpixel.height, 3)

    assert_equal(superpixel.count, 8)

    np_test.assert_array_equal(superpixel.image, image[0:3, 0:4])
    np_test.assert_array_equal(superpixel.mask, [
        [255, 255,   0, 255],
        [255, 255, 255, 255],
        [0,   0,   0,   255],
    ])

    assert_equals(superpixel.relative_center, (1.625, 0.75))
    assert_equals(superpixel.rounded_relative_center, (2, 1))
    assert_equals(superpixel.absolute_center, (1.625, 0.75))
    assert_equals(superpixel.rounded_absolute_center, (2, 1))
    assert_equals(superpixel.relative_center_in_bounding_box,
                  (1.625/4, 0.75/3))
    assert_equals(superpixel.mean, (46.25, 0.0, 0.0))

    assert_equals(superpixel.neighbors, set([2, 3, 4]))


def test_superpixel_2():
    superpixel = extract(image, superpixel_representation)[1]

    assert_equals(superpixel.id, 2)
    assert_equals(superpixel.order, 1)

    assert_equals(superpixel.left, 2)
    assert_equals(superpixel.top, 0)
    assert_equals(superpixel.width, 1)
    assert_equals(superpixel.height, 1)

    assert_equals(superpixel.count, 1)

    np_test.assert_array_equal(superpixel.image, image[0:1, 2:3])
    np_test.assert_array_equal(superpixel.mask, [
        [255]
    ])

    assert_equals(superpixel.relative_center, (0, 0))
    assert_equals(superpixel.rounded_relative_center, (0, 0))
    assert_equals(superpixel.absolute_center, (2, 0))
    assert_equals(superpixel.rounded_absolute_center, (2, 0))
    assert_equals(superpixel.relative_center_in_bounding_box, (0, 0))
    assert_equals(superpixel.mean, (20.0, 0.0, 0.0))

    assert_equals(superpixel.neighbors, set([1]))


def test_superpixel_3():
    superpixel = extract(image, superpixel_representation)[2]

    assert_equals(superpixel.id, 3)
    assert_equals(superpixel.order, 2)

    assert_equals(superpixel.left, 0)
    assert_equals(superpixel.top, 2)
    assert_equals(superpixel.width, 3)
    assert_equals(superpixel.height, 2)

    assert_equals(superpixel.count, 4)

    np_test.assert_array_equal(superpixel.image, image[2:4, 0:3])
    np_test.assert_array_equal(superpixel.mask, [
        [255, 255, 255],
        [255, 0,   0],
    ])

    assert_equals(superpixel.relative_center, (0.75, 0.25))
    assert_equals(superpixel.rounded_relative_center, (1, 0))
    assert_equals(superpixel.absolute_center, (0.75, 2.25))
    assert_equals(superpixel.rounded_absolute_center, (1, 2))
    assert_equals(superpixel.relative_center_in_bounding_box, (0.75/3, 0.25/2))
    assert_equals(superpixel.mean, (97.5, 0.0, 0.0))

    assert_equals(superpixel.neighbors, set([1, 4]))


def test_superpixel_4():
    superpixel = extract(image, superpixel_representation)[3]

    assert_equals(superpixel.id, 4)
    assert_equals(superpixel.order, 3)

    assert_equals(superpixel.left, 1)
    assert_equals(superpixel.top, 3)
    assert_equals(superpixel.width, 3)
    assert_equals(superpixel.height, 1)

    assert_equals(superpixel.count, 3)

    np_test.assert_array_equal(superpixel.image, image[3:4, 1:4])
    np_test.assert_array_equal(superpixel.mask, [
        [255, 255, 255]
        ])

    assert_equals(superpixel.relative_center, (1, 0))
    assert_equals(superpixel.rounded_relative_center, (1, 0))
    assert_equals(superpixel.absolute_center, (2, 3))
    assert_equals(superpixel.rounded_absolute_center, (2, 3))
    assert_equals(superpixel.relative_center_in_bounding_box, (1.0/3, 0))
    assert_equals(superpixel.mean, (140.0, 0.0, 0.0))

    assert_equals(superpixel.neighbors, set([1, 3]))
