from nose.tools import *
from numpy import testing as np_test

import numpy as np

from .helper import convert_1d_images_to_3d


def test_convert_1d_images_to_3d():
    array = np.array([range(2*3*4), range(2*3*4, 2*2*3*4)])

    converted = convert_1d_images_to_3d(array, width=3, height=4, depth=2)

    assert_equals(converted.shape, (2, 4, 3, 2))

    np_test.assert_equal(converted[0][0], [[0, 12], [1, 13], [2, 14]])
    np_test.assert_equal(converted[0][1], [[3, 15], [4, 16], [5, 17]])
    np_test.assert_equal(converted[0][2], [[6, 18], [7, 19], [8, 20]])
    np_test.assert_equal(converted[0][3], [[9, 21], [10, 22], [11, 23]])

    np_test.assert_equal(converted[1][0], [[24, 36], [25, 37], [26, 38]])
    np_test.assert_equal(converted[1][1], [[27, 39], [28, 40], [29, 41]])
    np_test.assert_equal(converted[1][2], [[30, 42], [31, 43], [32, 44]])
    np_test.assert_equal(converted[1][3], [[33, 45], [34, 46], [35, 47]])
