from nose.tools import *

import os

from .cifar10 import Cifar10


def test_download():
    # TODO: we get Travis errors cause log extends the allowed 4mb size
    # cifar10 = Cifar10('./datasets/cifar10')
    # assert_is_not_none(cifar10)
    # assert_true(os.path.exists('./datasets/cifar10'))
    pass
