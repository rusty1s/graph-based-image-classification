from nose.tools import *
from numpy import testing
from superpixel import Segment

image = [
        [0,   10,  20,  30],
        [40,  50,  60,  70],
        [80,  90,  100, 110],
        [120, 130, 140, 150],
        ]

superpixel = [
        [1, 1, 2, 1],
        [1, 1, 1, 1],
        [3, 3, 3, 1],
        [3, 4, 4, 4],
        ]


def test_generate():
    segments = Segment.generate(image, superpixel)

    assert_equal(len(segments), 4)


def test_segment_1():
    segment = Segment.generate(image, superpixel)[1]

    assert_equal(segment.left, 0)
    assert_equal(segment.top, 0)
    assert_equal(segment.right, 3)
    assert_equal(segment.bottom, 2)
    assert_equal(segment.width, 4)
    assert_equal(segment.height, 3)

    assert_equal(segment.count, 8)

    testing.assert_array_equal(segment.image, [
        [0,   10,  20,  30],
        [40,  50,  60,  70],
        [80,  90,  100, 110],
        ])
    testing.assert_array_equal(segment.mask, [
        [255, 255,   0, 255],
        [255, 255, 255, 255],
        [0,   0,   0,   255],
        ])

    assert_equal(segment.center, (1.625, 0.75))
    assert_equal(segment.mean[0], 46.25)


def test_segment_2():
    segment = Segment.generate(image, superpixel)[2]

    assert_equal(segment.left, 2)
    assert_equal(segment.top, 0)
    assert_equal(segment.right, 2)
    assert_equal(segment.bottom, 0)
    assert_equal(segment.width, 1)
    assert_equal(segment.height, 1)

    assert_equal(segment.count, 1)

    testing.assert_array_equal(segment.image, [
        [20]
        ])
    testing.assert_array_equal(segment.mask, [
        [255]
        ])

    assert_equal(segment.center, (2, 0))
    assert_equal(segment.mean[0], 20)


def test_segment_3():
    segment = Segment.generate(image, superpixel)[3]

    assert_equal(segment.left, 0)
    assert_equal(segment.top, 2)
    assert_equal(segment.right, 2)
    assert_equal(segment.bottom, 3)
    assert_equal(segment.width, 3)
    assert_equal(segment.height, 2)

    assert_equal(segment.count, 4)

    testing.assert_array_equal(segment.image, [
        [80,  90,  100],
        [120, 130, 140],
        ])
    testing.assert_array_equal(segment.mask, [
        [255, 255, 255],
        [255, 0,   0],
        ])

    assert_equal(segment.center, (0.75, 2.25))
    assert_equal(segment.mean[0], 97.5)


def test_segment_4():
    segment = Segment.generate(image, superpixel)[4]

    assert_equal(segment.left, 1)
    assert_equal(segment.top, 3)
    assert_equal(segment.right, 3)
    assert_equal(segment.bottom, 3)
    assert_equal(segment.width, 3)
    assert_equal(segment.height, 1)

    assert_equal(segment.count, 3)

    testing.assert_array_equal(segment.image, [
        [130, 140, 150],
        ])
    testing.assert_array_equal(segment.mask, [
        [255, 255, 255]
        ])

    assert_equal(segment.center, (2, 3))
    assert_equal(segment.mean[0], 140)
