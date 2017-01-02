import abc
import six


@six.add_metaclass(abc.ABCMeta)
class DataSet():

    @property
    @abc.abstractmethod
    def name(self):
        pass

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

    @abc.abstractmethod
    def read(self, filename_queue):
        pass

    def train_preprocess(self, record):
        return record

    def eval_preprocess(self, record):
        return record
