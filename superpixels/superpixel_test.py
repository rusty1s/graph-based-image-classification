from nose.tools import *
from numpy import testing as np_test

import numpy as np

from .superpixel import Superpixel


def test_non_initialised_superpixel():
    superpixel = Superpixel()

    assert_is_none(superpixel.id)
    assert_is_none(superpixel.order)

    assert_equal(superpixel.left, 0)
    assert_equal(superpixel.top, 0)

    assert_equal(superpixel.neighbors, set())
    np_test.assert_equal(superpixel.image, np.empty((0, 0, 3), dtype=np.float))
    np_test.assert_equal(superpixel.mask, np.empty((0, 0, 1), dtype=np.uint8))

    assert_equal(superpixel.width, 0)
    assert_equal(superpixel.height, 0)
    assert_equal(superpixel.count, 0)

    assert_true(superpixel.relative_center, (0, 0))
    assert_true(superpixel.rounded_relative_center, (0, 0))
    assert_true(superpixel.absolute_center, (0, 0))
    assert_true(superpixel.rounded_absolute_center, (0, 0))
    assert_equals(superpixel.relative_center_in_bounding_box, (0, 0))
    assert_equal(superpixel.mean, (0.0, 0.0, 0.0))


def test_initialised_superpixel():
    image = np.array([
            [[15.0, 30.0, 45.0], [30.0, 45.0, 60.0], [45.0, 60.0, 75.0]],
            [[60.0, 75.0, 90.0], [75.0, 90.0, 105.0], [90.0, 105.0, 120.0]],
        ])

    mask = np.array([
            [255, 255, 255],
            [255,  0, 0],
        ], dtype=np.uint8)

    superpixel = Superpixel(id=0, order=1, left=10, top=20,
                            neighbors=set([2, 3]), image=image, mask=mask)

    assert_equals(superpixel.id, 0)
    assert_equals(superpixel.order, 1)

    assert_equals(superpixel.left, 10)
    assert_equals(superpixel.top, 20)

    assert_equals(superpixel.neighbors, set([2, 3]))
    np_test.assert_equal(superpixel.image, image)
    np_test.assert_equal(superpixel.mask, mask)

    assert_equals(superpixel.width, 3)
    assert_equals(superpixel.height, 2)
    assert_equals(superpixel.count, 4)

    assert_equals(superpixel.relative_center, (0.75, 0.25))
    assert_equals(superpixel.rounded_relative_center, (1, 0))
    assert_equals(superpixel.absolute_center, (10.75, 20.25))
    assert_equals(superpixel.rounded_absolute_center, (11, 20))
    assert_equals(superpixel.relative_center_in_bounding_box, (0.75/3, 0.25/2))
    assert_equals(superpixel.mean, (37.5, 52.5, 67.5))
