from nose.tools import *

from .convolutional_1d import (output_width)


def test_output_width():
    pass
    # The convolution and max pooling operations have padding set to 'SAME', so
    # that we have always no-zero padded outputs.

    # We have an input width of 7, a patch size of 3 and a stride size of 2.
    # That means after convolution our input width is reduced to 3. A pooling
    # of stride 1 shouldn't change this.
    width = output_width(7, 3, 2, 1)
    assert_equals(width, 3)

    # We have an input width of 10, a patch size of 2 and a stride size of 1.
    # That means after convolution our input width is reduced to 9. A pooling
    # of stride 2 should reduce this to 4.
    width = output_width(10, 2, 1, 2)
    assert_equals(width, 4)

    # We have an input width of 10, a patch size of 1 and a stride size of 4.
    # That means after convolution our input width is reduced to 2. A pooling
    # of stride 2 should reduce this to 1.
    width = output_width(10, 1, 4, 2)
    assert_equals(width, 1)
