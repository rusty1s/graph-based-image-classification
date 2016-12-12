from nose.tools import *
from numpy import testing as np_test

import numpy as np

from .model import convert_input


def test_convert_input():
    array = np.array([
        [
            [1,  2,  3,  4],
            [5,  6,  7,  8],
            [9, 10, 11, 12],
        ],
        [
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ],
    ])

    array = convert_input(array)

    np_test.assert_equal(array, [
        [1,  13],
        [2,  14],
        [3,  15],
        [4,  16],
        [5,  17],
        [6,  18],
        [7,  19],
        [8,  20],
        [9,  21],
        [10, 22],
        [11, 23],
        [12, 24],
    ])
