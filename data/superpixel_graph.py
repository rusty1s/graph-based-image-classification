from .dataset import DataSet


class GraphDataSet(DataSet):

    def __init__(self, dataset, grapher, normalizer):
        self._dataset = dataset

        self._grapher = grapher
        self._normalizer = normalizer

    @property
    def data_dir(self):
        return self.dataset.data_dir

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
        return normalizer.shape

    def read(self, filename_queue):
        return self._dataset.read(filename_queue)

    def train_preprocess(self, image):
        image = self._dataset.train_preprocess(image)
        return self.postprocess(image)

    def eval_preprocess(self, image):
        image = self._dataset.eval_preprocess(image)
        return self.postprocess(image)

    def _postprocesss(self, image):
        graph = self._to_graph(image)
        self._normalizer.graph = graph
        return self._normalizer.normalize()
