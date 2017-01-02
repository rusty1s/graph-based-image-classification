import os

import tensorflow as tf

from grapher import SuperpixelGrapher
from superpixel.algorithm import slico_generator

from .dataset import DataSet
from .io import (write, read_and_decode)
from .record import Record


class PatchySanDataSet(DataSet):

    def __init__(self, dataset, data_dir=None, train_epochs=1):
        if not data_dir:
            data_dir = '/tmp/{}_patchy_san'.format(dataset.name)

        self._dataset = dataset
        self._data_dir = data_dir
        self._grapher = SuperpixelGrapher(slico_generator(25))

        if tf.gfile.Exists(data_dir):
            tf.gfile.DeleteRecursively(data_dir)
        tf.gfile.MakeDirs(data_dir)

        # Write evaluation dataset.
        write(dataset.num_examples_per_epoch_for_eval, dataset.eval_filenames,
              self.eval_filenames[0], dataset.read,
              preprocess=self.eval_convert, epochs=1, eval_data=True,
              dataset_name=self.name)

    @property
    def name(self):
        return 'Patchy San ({})'.format(self._dataset.name)

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def train_filenames(self):
        return [os.path.join(self.data_dir, 'train.tfrecords')]

    @property
    def eval_filenames(self):
        return [os.path.join(self.data_dir, 'eval.tfrecords')]

    @property
    def num_labels(self):
        return self._dataset.num_labels

    @property
    def num_examples_per_epoch_for_train(self):
        return self._dataset.num_examples_per_epoch_for_train

    @property
    def num_examples_per_epoch_for_eval(self):
        return self._dataset.num_examples_per_epoch_for_eval

    def read(self, filename_queue):
        return read_and_decode(filename_queue, 20, 1, 8)

    def train_convert(self, record):
        record = self._dataset.train_preprocess(record)
        return self.convert(record)

    def eval_convert(self, record):
        record = self._dataset.eval_preprocess(record)
        return self.convert(record)

    def convert(self, record):
        nodes, adjacent = self._grapher.create_graph(record.data)
        nodes = tf.reshape(nodes, [-1, 1, 8])
        nodes = tf.strided_slice(nodes, [0], [20], [1])

        return Record(20, 1, 8, record.label, nodes)
