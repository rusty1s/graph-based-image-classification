import os
import numpy as np
import cv2

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle

# from .helper import (path_exists, extract_tar, download as download_tar,
#                      convert_batch)

# The CIFAR-10 dataset consists of 60.000 32x32 colour images in 10 classes,
# with 6.000 images per class. There are 50.000 training images and 10.000 test
# images.
# The dataset is divided into five training batches and one test batch, each
# with 10.000 images.

NUM_TRAIN_BATCHES = 5

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
#    stores a 32x32 colour image. The first 1024 entries contain the red
#    channel values, the next 1024 the green, and the final 1024 the blue. The
#    image is stored in row-major order, so that the first 32 entries of the
#    array are the red channel values of the first row of the image.
# 2. **labels** -- a list of 10.000 numbers in the range of 0-9. The number at
#    index i indicates the label of the ith image in the array `data`.


# The dataset contains another file, called `batches.meta`. It too contains a
# Python dictionary object. It has the following entries:
# 1. **label_names** -- a 10-element list which gives meaningful names to the
#    numeric labels in the `labels` array desribed above. For example,
#    `label_names[0] == 'airplane'`, etc.


class Cifar10(object):
    def __init__(self, dir):
        self.__dir = dir

        self.download()

    # @property
    # def dir(self):
    #     return self.__dir

    # @property
    # def label_names(self):
    #     """A 10-element list which gives meaningful names to the numeric
    #     labels, e.g. `label_names[0] == 'airplane'`."""

    #     with open(os.path.join(self.dir, 'batches.meta'), 'rb') as f:
    #         return pickle.load(f)['label_names']

    # @property
    # def num_labels(self):
    #     """Returns the number of labels."""

    #     return len(self.label_names)

    # def download(self):
    #     """Downloads the CIFAR-10 dataset, extracts it and moves it to
    #     `self.dir`."""

    #     if path_exists(self.dir, 'downloading'):
    #         print('Everything is fine if {} already contains the CIFAR-10 '
    #               'dataset.'.format(self.dir))
    #         return

    #     download_tar(URL, TAR_NAME)
    #     extract_tar(TAR_NAME, self.dir)
    #     os.remove(os.path.join(self.dir, 'readme.html'))

    #     for i in range(NUM_TRAIN_BATCHES):
    #         file = os.path.join(self.dir, 'data_batch_{}'.format(i+1))
    #         convert_batch(file, self.num_labels)

    #     convert_batch(os.path.join(self.dir, 'test_batch'), self.num_labels)

    # def get_train_batch(self, batch_num):
    #     """Gets the nth training batch (0 <= n < 5) and returns a dictionary
    #     containing 10.000th labels and 3d images."""

    #     if not 0 <= batch_num < NUM_TRAIN_BATCHES:
    #         raise ValueError('Invalid batch number. Batch number must be '
    #                          'between 0 and 4.')

    #     filename = 'data_batch_{}'.format(batch_num + 1)
    #     path = os.path.join(self.dir, filename)

    #     with open(path, 'rb') as f:
    #         return pickle.load(f)

    # def get_test_batch(self):
    #     """Gets the test batch and returns a dictionary containing 10.000th
    #     labels and 3d images."""

    #     with open(os.path.join(self.dir, 'test_batch'), 'rb') as f:
    #         return pickle.load(f)

    # def save_images(self, dir=None):
    #     """Saves all images to the `dir` directory. Train images go to
    #     `dir/train`. Test images go to `dir/test`. Images go to its
    #     corresponding label directory and are named incrementally."""

    #     # Set fallback directory.
    #     if dir is None:
    #         dir = os.path.join(self.dir, 'images')

    #     train_dir = os.path.join(dir, 'train')
    #     test_dir = os.path.join(dir, 'test')

    #     # Abort if any of the needed directories already exists
    #     if path_exists(train_dir, 'saving images'):
    #         return

    #     if path_exists(test_dir, 'saving images'):
    #         return

    #     # Create the folder structure.
    #     for label in self.label_names:
    #         os.makedirs(os.path.join(train_dir, label))
    #         os.makedirs(os.path.join(test_dir, label))

    #     # Create two dictionaries that save the current file index for each
    #     # label.
    #     train_indices = {label: 1 for label in self.label_names}
    #     test_indices = train_indices.copy()

    #     # Save the train images to train directory.
    #     for batch_num in range(0, NUM_TRAIN_BATCHES):
    #         batch = self.get_train_batch(batch_num)
    #         self.__save_images_from_batch(batch, train_indices, train_dir)

    #     # Save the test images to the test directory.
    #     test_batch = self.get_test_batch()
    #     self.__save_images_from_batch(test_batch, test_indices, test_dir)

    # def __save_images_from_batch(self, batch, indices, dir):
    #     """Saves all images of a batch to the `dir` directory. Images go to
    #     its corresponding label directory and are named incrementally."""

    #     for i in range(0, len(batch['labels'])):
    #         label_index = np.argmax(batch['labels'][i])
    #         label = self.label_names[label_index]
    #         data = batch['data'][i]

    #         filename = '{}.png'.format(indices[label])
    #         file = os.path.join(dir, label, filename)

    #         cv2.imwrite(file, data)

    #         # Increment the label index.
    #         indices[label] += 1
