from __future__ import print_function

import os
import requests
from clint.textui import progress
import tarfile

# load the correct pickle implementation for different python versions
try:
    import cPickle as pickle
except:
    import _pickle as pickle

# The CIFAR-10 dataset consists of 60.000 32x32 colour images in 10 classes,
# with 6.000 images per class. There are 50.000 training images and 10.000 test
# images.
# The dataset is divided into five training batches and one test bach, each
# with 10.000 images.

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
NUM_CLASSES = 10
NUM_TRAIN_IMAGES = 50000
NUM_TEST_IMAGES = 10000
BATCH_SIZE = 10000

# **Classes:**
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck
#
# The classes are completely mutually exclusive. There is no overlap between
# automobiles and trucks. "Automobile" includes sedans, SUVs, things of that
# sort. "Truck" includes only big trucks. Neither includes pickup trucks.

TAR_NAME = 'cifar-10-python.tar.gz'
URL = 'http://www.cs.toronto.edu/~kriz/' + TAR_NAME

# The archive contains the files `data_batch_1`, `data_batch_2`, ...,
# `data_batch_5`, as well as `test_batch`. Each of these files is a Python
# "pickled" object produced with `cPickle`.
#
# Each of the batch files contains a dictionary with the following elements:
# 1. **data** -- a 10.000x3072 `numpy` array of `uint8`s. Each row of the array
#    i stores a 32x32 colour image. The first 1024 entries contain the red
#    channel values, the next 1024 the green, and the final 1024 the blue. The
#    image is stored in row-major order, so that the first 32 entries of the
#    array are the red channel values of the first row of the image.
# 2. **labels** -- a list of 10.000 numbers in the range of 0-9. The number at
#    index i indicates the label of the ith image in the array `data`.

IMAGE_SIZE = 3072

# The dataset contains another file, called `batches.meta`. It too contains a
# Python dictionary object. It has the following entries:
# 1. **label_names** -- a 10-element list which gives meaningful names to the
#    numeric labels in the `labels` array desribed above. For example,
#    `label_names[0] == 'airplane'`, etc.


class Cifar10(object):
    def __init__(self, dir):
        self.dir = dir

        self.download(dir)

    def download(self, dir):
        """Downloads the CIFAR-10 dataset, extracts it and moves it to `dir`.
        """

        if os.path.exists(dir):
            print('Specified directory already exists. Does it contain the '
                  'CIFAR-10 dataset? Skip downloading.')
            return

        print('Downloading CIFAR-10 dataset. This can take a while...')

        r = requests.get(URL, stream=True)

        # print a pretty progress bar while downloading
        with open(TAR_NAME, 'wb') as f:
            total_length = r.headers.get('content-length')
            for chunk in progress.bar(r.iter_content(chunk_size=1024),
                                      expected_size=(int(total_length)/1024)):
                if chunk:
                    f.write(chunk)
                    f.flush()

        print('Unpacking CIFAR-10 dataset.')

        with tarfile.open(TAR_NAME, 'r:gz') as tar:
            extracted_dir = tar.getnames()[0]
            tar.extractall()

        print('Moving CIFAR-10 dataset to {}.'.format(dir))

        os.makedirs(dir)
        os.rename(extracted_dir, dir)

        # remove tar file
        os.remove(TAR_NAME)
