from .dataset import DataSet


# IDEA:
# Curry the superpixel algorithm, so that it takes a single argument: image
# Curry the graph algorithm, so that it takes a single argument: the superpixel
# represenation
class SuperpixelGraph(DataSet):

    def __init__(self, dataset):
        self._dataset = dataset

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
        # TODO
        return None

    def read(self, filename_queue):
        return self._dataset.read(filename_queue)

    def train_preprocess(self, image):
        image = self._dataset.train_preprocess(image)

        data = image

        # TODO

        return data

    def eval_preprocess(self, image):
        image = self._dataset.eval_preprocess(image)

        data = image

        # TODO

        return data
