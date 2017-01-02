from .dataset import DataSet


class ConvertedDataSet(DataSet):

    def __init__(self, dataset, converter):
        self._dataset = dataset
        self._converter = converter(dataset)

    @property
    def data_dir(self):
        return self._converter.data_dir

    @property
    def train_filenames(self):
        return self._converter.train_filenames

    @property
    def eval_filenames(self):
        return self._converter.eval_filenames

    @property
    def num_labels(self):
        return self._dataset.num_labels

    @property
    def num_examples_per_epoch_for_train(self):
        return self._dataset.num_examples_per_epoch_for_train

    @property
    def num_examples_per_epoch_for_eval(self):
        return self._dataset.num_examples_per_epoch_for_eval

    @property
    def data_shape(self):
        return self._converter.data_shape

    @property
    def read(self, filename_queue):
        return self._converter.read(filename_queue)
