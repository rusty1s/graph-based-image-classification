from nose.tools import *
from numpy import testing
from superpixel import Segment


def test_generate():
    superpixel = [
            [1, 1, 2, 1],
            [1, 1, 1, 1],
            [3, 3, 3, 1],
            [3, 4, 4, 4],
            ]
    segments = Segment.generate(superpixel)

    assert_equal(len(segments), 4)

    assert_equal(segments[1].left, 0)
    assert_equal(segments[1].top, 0)
    assert_equal(segments[1].width, 4)
    assert_equal(segments[1].height, 3)
    testing.assert_array_equal(segments[1].mask, [
        [255, 255,   0, 255],
        [255, 255, 255, 255],
        [0,   0,   0,   255],
        ])
    assert_equal(segments[1].center, (1.625, 0.75))

    assert_equal(segments[2].left, 2)
    assert_equal(segments[2].top, 0)
    assert_equal(segments[2].width, 1)
    assert_equal(segments[2].height, 1)
    testing.assert_array_equal(segments[2].mask, [[255]])
    assert_equal(segments[2].center, (0, 0))

    assert_equal(segments[3].left, 0)
    assert_equal(segments[3].top, 2)
    assert_equal(segments[3].width, 3)
    assert_equal(segments[3].height, 2)
    testing.assert_array_equal(segments[3].mask, [
        [255, 255, 255],
        [255, 0,   0],
        ])
    assert_equal(segments[3].center, (0.75, 0.25))

    assert_equal(segments[4].left, 1)
    assert_equal(segments[4].top, 3)
    assert_equal(segments[4].width, 3)
    assert_equal(segments[4].height, 1)
    testing.assert_array_equal(segments[4].mask, [[255, 255, 255]])
    assert_equal(segments[4].center, (1, 0))
