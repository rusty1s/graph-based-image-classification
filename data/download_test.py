from nose.tools import *

import os

from .download import maybe_download_and_extract


def test_maybe_download_and_extract():
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    maybe_download_and_extract(url, data_dir='/tmp', dirname='cifar-10',
                               show_progress=False)

    path = os.path.join('/tmp/cifar-10')
    assert_true(os.path.exists(path))
    assert_true(os.path.exists(os.path.join(path, 'data_batch_1')))
    assert_true(os.path.exists(os.path.join(path, 'data_batch_2')))
    assert_true(os.path.exists(os.path.join(path, 'data_batch_3')))
    assert_true(os.path.exists(os.path.join(path, 'data_batch_4')))
    assert_true(os.path.exists(os.path.join(path, 'data_batch_5')))
    assert_true(os.path.exists(os.path.join(path, 'test_batch')))
    assert_true(os.path.exists(os.path.join(path, 'batches.meta')))

    # Shouldn't do anything if folder already exists.
    open(os.path.join(path, 'test'), 'a').close()
    assert_true(os.path.exists(os.path.join(path, 'test')))

    maybe_download_and_extract(url, data_dir='/tmp', dirname='cifar-10')
    assert_true(os.path.exists(path))
    assert_true(os.path.exists(os.path.join(path, 'test')))
    os.remove(os.path.join(path, 'test'))
