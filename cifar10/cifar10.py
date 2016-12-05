from __future__ import print_function

import os
import shutil
import requests
from clint.textui import colored, progress
import tarfile
import cv2
import numpy as np

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
BATCH_COUNT = 5
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

        self.download()

        # set the label_names by reading `batches.meta`
        with open(os.path.join(self.dir, 'batches.meta'), 'rb') as f:
            self.__label_names = pickle.load(f)['label_names']

    @property
    def label_names(self):
        """A 10-element list which gives meaningful names to the numeric
        labels, e.g. `label_names[0] == 'airplane'`."""

        return self.__label_names

    def download(self):
        """Downloads the CIFAR-10 dataset, extracts it and moves it to
        `self.dir`."""

        if os.path.exists(self.dir):
            print(colored.red('Abort downloading: '
                  '{} already exists.'.format(self.dir)))
            print('Everything is fine if {} already contains the CIFAR-10 '
                  'dataset.'.format(self.dir))
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

        print('Moving CIFAR-10 dataset to {}.'.format(self.dir))

        os.makedirs(self.dir)
        shutil.move(extracted_dir, self.dir)

        # remove tar file
        os.remove(TAR_NAME)

    def get_train_batch(self, batch_num):
        if not 0 <= batch_num < BATCH_COUNT:
            raise ValueError('Invalid batch number. Batch number must be '
                             'between 0 and 4.')

        filename = 'data_batch_{}'.format(batch_num + 1)
        path = os.path.join(self.dir, filename)

        with open(path, 'rb') as f:
            # encode to latin1 to avoid `UnicodeDecodeError`
            batch = pickle.load(f, encoding='latin1')

        return batch

    def get_test_batch(self):
        with open(os.path.join(self.dir, 'test_batch'), 'rb') as f:
            # encode to latin1 to avoid `UnicodeDecodeError`
            batch = pickle.load(f, encoding='latin1')

        return batch

    def data_to_image(self, data):
        # convert the image_data to an actual 3d-array image
        c_len = int(IMAGE_SIZE/3)  # channel length

        red = data[0:c_len]
        green = data[c_len:2 * c_len]
        blue = data[2 * c_len:3 * c_len]

        return np.dstack((red, green, blue)).reshape(IMAGE_WIDTH, IMAGE_HEIGHT)

    def save_images(self):
        """Saves all images to the `self.dir` directory.
        Train images go to `self.dir/train`. Test images go to `self.dir/test`.
        Images go to its corresponding label directory and are named
        incrementally."""

        # remove already saved train images
        try:
            shutil.rmtree(os.path.join(self.dir, 'train'))
        except OSError:
            pass

        # remove already saved test images
        try:
            shutil.rmtree(os.path.join(self.dir, 'test'))
        except OSError:
            pass

        # create the necessary directories
        for label in self.label_names:
            os.makedirs(os.path.join(self.dir, 'train', label))
            os.makedirs(os.path.join(self.dir, 'test', label))

        # create to dictionaries that save the current filename for each label
        train_numbers = {label: 0 for label in self.label_names}
        test_numbers = {label: 0 for label in self.label_names}

        test_batch = self.get_test_batch()
        for i in range(0, NUM_TEST_IMAGES):
            label = self.label_names[test_batch['labels'][i]]
            image_data = test_batch['data'][i]

            filename = '{}.png'.format(test_numbers[label])
            file = os.path.join(self.dir, 'test', label, filename)

            image = self.data_to_image(image_data)

            cv2.imwrite(file, image)

            # increment the filename
            test_numbers[label] += 1

        for batch_num in range(0, BATCH_COUNT):
            train_batch = self.get_train_batch(batch_num)

            for i in range(0, BATCH_SIZE):
                label = self.label_names[train_batch['labels'][i]]
                image_data = train_batch['data'][i]

                filename = '{}.png'.format(train_numbers[label])
                file = os.path.join(self.dir, 'train', label, filename)

                image = self.data_to_image(image_data)

                cv2.imwrite(file, image)

                # increment the filename
                train_numbers[label] += 1
