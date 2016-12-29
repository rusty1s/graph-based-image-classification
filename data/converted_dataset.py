from .dataset import DataSet


class ConvertedDataSet(DataSet):

    def __init__(self, dataset, converter):
        self._dataset = dataset
        self._converter = converter

    @property
    def data_dir(self):
        return self._dataset.data_dir

    @property
    def train_filenames(self):
        return self._dataset.train_filenames

    @property
    def eval_filenames(self):
        return self._dataset.eval_filenames

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
        return self._converter.shape

    def read(self, filename_queue):
        return self._dataset.read(filename_queue)

    def train_preprocess(self, data):
        data = self._dataset.train_preprocess(data)
        return self._postprocess(data)

    def eval_preprocess(self, data):
        data = self._dataset.eval_preprocess(data)
        return self._postprocess(data)

    def _postprocess(self, data):
        return self._converter.convert(data)
