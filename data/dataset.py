import abc
import six


@six.add_metaclass(abc.ABCMeta)
class DataSet():

    @property
    @abc.abstractmethod
    def data_dir(self):
        pass

    @property
    @abc.abstractmethod
    def train_filenames(self):
        pass

    @property
    @abc.abstractmethod
    def eval_filenames(self):
        pass

    @property
    @abc.abstractmethod
    def num_labels(self):
        pass

    @property
    @abc.abstractmethod
    def num_examples_per_epoch_for_train(self):
        pass

    @property
    @abc.abstractmethod
    def num_examples_per_epoch_for_eval(self):
        pass

    @property
    @abc.abstractmethod
    def data_shape(self):
        pass

    @abc.abstractmethod
    def read(self, filename_queue):
        pass

    def train_preprocess(self, data):
        return data

    def eval_preprocess(self, data):
        return data


# import os
# import numpy as np

# # Load the correct pickle implementation for different python versions.
# try:
#     import cPickle as pickle
# except:
#     import _pickle as pickle


# def load_dataset(base_dir, names):
#     batches = []

#     for name in names:
#         path = os.path.join(base_dir, name)
#         with open(path, 'rb') as f:
#             batches += [pickle.load(f)]

#     return Dataset(batches)


# class Dataset(object):

#     def __init__(self, batches):
#         """Constructs a Dataset."""

#         self._data = np.concatenate([batch['data'] for batch in batches])
#         self._labels = np.concatenate([batch['labels'] for batch in batches])

#         self._epochs_completed = 0
#         self._index_in_epoch = 0

#     @property
#     def data(self):
#         return self._data

#     @property
#     def labels(self):
#         return self._labels

#     @property
#     def num_examples(self):
#         return self.data.shape[0]

#     @property
#     def epochs_completed(self):
#         return self._epochs_completed

#     def add_batch(self, batch):
#         self._data = np.concatenate([self.data, batch['data']])
#         self._labels = np.concatenate([self.labels, batch['labels']])

#     def next_batch(self, batch_size):
#         """Returns the next `batch_size` examples from this data set."""

#         assert batch_size <= self.num_examples

#         start = self._index_in_epoch
#         self._index_in_epoch += batch_size

#         if self._index_in_epoch > self.num_examples:
#             # Finished epoch.
#             self._epochs_completed += 1

#             # Shuffle the data.
#             perm = np.arange(self.num_examples)
#             np.random.shuffle(perm)

#             self.__data = self.data[perm]
#             self.__labels = self.labels[perm]

#             # Start next epoch.
#             start = 0
#             self._index_in_epoch = batch_size

#         end = self._index_in_epoch

#         return self.data[start:end], self.labels[start:end]
