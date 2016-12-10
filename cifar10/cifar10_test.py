from nose.tools import *

import os

from .cifar10 import Cifar10


def test_download():
    cifar10 = Cifar10('./datasets/cifar10')
    assert_is_not_none(cifar10)

    assert_true(os.path.exists('./datasets/cifar10'))
