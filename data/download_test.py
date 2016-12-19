from nose.tools import *

from .download import maybe_download_and_extract


def test_maybe_download_and_extract():
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    maybe_download_and_extract(url, data_dir='/tmp', dirname='cifar-10')
