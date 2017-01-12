import abc
import six


@six.add_metaclass(abc.ABCMeta)
class DataSet():
    """Abstract class for defining a dataset interface."""

    @property
    @abc.abstractmethod
    def data_dir(self):
        """The path to the directory where the dataset is stored.

        Returns:
            A string with the path to the dataset.
        """
        pass

    @property
    @abc.abstractmethod
    def train_filenames(self):
        """The filenames of the training batches from the dataset.

        Returns:
             A list of filenames.
        """
        pass

    @property
    @abc.abstractmethod
    def eval_filenames(self):
        """The filenames of the evaluating batches from the dataset.

        Returns:
             A list of filenames.
        """
        pass

    @property
    @abc.abstractmethod
    def labels(self):
        """The ordered labels of the dataset.

        Returns:
            A list of labels.
        """
        pass

    @property
    def num_labels(self):
        """The number of labels of the dataset.

        Return:
            A number.
        """
        return len(self.labels)

    def label_index(self, label_name):
        """The index of the given label.

        Args:
            label: The label.

        Returns:
            A number.

        Raises:
            ValueError: If the label cannot be found in the labels.
        """

        index = self.labels.index(label_name)

        if index == -1:
            raise ValueError('{} is no valid label name.'.format(label_name))

        return index

    def label_name(self, index):
        """The label name of the given index.

        Args:
            index: The index:

        Returns:
            A string describing the label.

        Raises:
            ValueError: If the label of the index cannot be found in the
              labels.
        """

        if 0 <= index < self.num_labels:
            return self.labels[index]

        raise ValueError('{} is no valid label index.'.format(index))

    @property
    @abc.abstractmethod
    def num_examples_per_epoch_for_train(self):
        """The number of examples per epoch for training the dataset.

        Return:
            A number.
        """
        pass

    @property
    @abc.abstractmethod
    def num_examples_per_epoch_for_eval(self):
        """The number of examples per epoch for evaluating the dataset.

        Return:
            A number.
        """
        pass

    @abc.abstractmethod
    def read(self, filename_queue):
        """Reads and parses examples from data files.

        Args:
            filename_queue: A queue of strings with the filenames to read from.

        Returns:
            A record object.
        """
        pass
