import numpy as np


class DataSet(object):

    def __init__(self, batches):
        """Constructs a DataSet."""

        assert len(batches) > 0
        assert 'data' in batches[0]
        assert 'labels' in batches[0]

        self.__num_examples = 0
        for batch in batches:
            self.__num_examples += batch['data'].shape[0]

        self.__epochs_completed = 0
        self.__index_in_epoch = 0

        data_shape = list(batches[0]['data'].shape)
        labels_shape = list(batches[0]['labels'].shape)
        data_shape[0] = self.__num_examples
        labels_shape[0] = self.__num_examples

        self.__data = np.zeros(data_shape)
        self.__labels = np.zeros(labels_shape)

        i = 0
        j = 0
        for batch in batches:
            for data in batch['data']:
                self.__data[i] = data
                i += 1

            for label in batch['labels']:
                self.__labels[j] = label
                j += 1

    @property
    def data(self):
        return self.__data

    @property
    def labels(self):
        return self.__labels

    @property
    def num_examples(self):
        return self.__num_examples

    @property
    def epochs_completed(self):
        return self.__epochs_completed

    def next_batch(self, batch_size):
        """Returns the next `batch_size` examples from this data set."""

        assert batch_size <= self.__num_examples

        start = self.__index_in_epoch
        self.__index_in_epoch += batch_size

        if self.__index_in_epoch > self.__num_examples:
            # Finished epoch.
            self.__epochs_completed += 1

            # Shuffle the data.
            perm = np.arange(self.__num_examples)
            np.random.shuffle(perm)

            self.__data = self.__data[perm]
            self.__labels = self.__labels[perm]

            # Start next epoch.
            start = 0
            self.__index_in_epoch = batch_size

        end = self.__index_in_epoch

        return self.__data[start:end], self.__labels[start:end]
