import abc
import six


@six.add_metaclass(abc.ABCMeta)
class DataSet():
    """Abstract class for defining a dataset interface."""

    @property
    @abc.abstractmethod
    def name(self):
        """The name of the dataset for pretty printing.

        Returns:
            A String with the name of the dataset.
        """
        pass

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
    def classes(self):
        """The ordered classes of the dataset.

        Returns:
            A list of classes.
        """
        pass

    @property
    def num_classes(self):
        """The number of classes of the dataset.

        Return:
            A number.
        """
        return len(self.classes)

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
