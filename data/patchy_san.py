import os

import tensorflow as tf
import numpy as np

from .dataset import DataSet
from .io import (write, read_and_decode)
from .record import Record

from superpixels import create_superpixel_graph
from superpixels import image_to_slic_zero
from superpixels import extract_superpixels
from patchy import order
from patchy import betweenness_centrality
from patchy import receptive_fields


class PatchySanDataSet(DataSet):

    def __init__(self, dataset, data_dir=None, train_epochs=1):
        if not data_dir:
            data_dir = '/tmp/{}_patchy_san'.format(dataset.name)

        self._dataset = dataset
        self._data_dir = data_dir
        # self._grapher = SuperpixelGrapher(slico_generator(100))

        if tf.gfile.Exists(data_dir):
            tf.gfile.DeleteRecursively(data_dir)
        tf.gfile.MakeDirs(data_dir)

        # Write evaluation dataset.
        # write(dataset.num_examples_per_epoch_for_train, dataset.train_filenames,
        #       self.train_filenames[0], dataset.read,
        #       preprocess=self.train_convert, epochs=1, eval_data=False,
        #       dataset_name=self.name)
        # write(dataset.num_examples_per_epoch_for_eval, dataset.eval_filenames,
        #       self.eval_filenames[0], dataset.read,
        #       preprocess=self.eval_convert, epochs=1, eval_data=True,
        #       dataset_name=self.name)

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
        return read_and_decode(filename_queue, 25, 10, 8)

    def train_convert(self, record):
        record = self._dataset.train_preprocess(record)
        return self.convert(record)

    def eval_convert(self, record):
        record = self._dataset.eval_preprocess(record)
        return self.convert(record)

    def convert(self, record):
        def convert_py(image):
            def node_map(superpixel):
                return superpixel.get_attributes()

            def edge_map(from_superpixel, to_superpixel):
                return {'weight': 1}

            def node_features(node_attributes):
                return [
                    node_attributes['red'],
                    node_attributes['green'],
                    node_attributes['blue'],
                    node_attributes['y'],
                    node_attributes['x'],
                    node_attributes['count'],
                    node_attributes['height'],
                    node_attributes['width'],
                ]

            image = image.astype(np.int32)
            segmentation = image_to_slic_zero(image, 50)
            superpixels = extract_superpixels(image, segmentation)
            graph = create_superpixel_graph(superpixels, node_map, edge_map)

            conv = receptive_fields(graph, order, 2, 25, 10,
                                    betweenness_centrality, node_features, 8)
            return conv.astype(np.float32)

        conv = tf.py_func(convert_py, [record.data], tf.float32)
        return Record(25, 10, 8, record.label, conv)
        # nodes, adjacent = self._grapher.create_graph(record.data)

        # sorted_node_indices = labelings['betweenness_centrality'](adjacent)
        # sorted_node_indices = tf.strided_slice(sorted_node_indices, [0], [100], [1])
        # sorted_node_indices = tf.cast(sorted_node_indices, tf.float32)
        # sorted_node_indices = tf.reshape(sorted_node_indices, [1, 1, 100])
        # return Record(1, 1, 100, record.label, sorted_node_indices)

        # adjacent = tf.strided_slice(adjacent, [0, 0], [10, 10], [1, 1])
        # adjacent = tf.reshape(adjacent, [10, 10, 1])
        # print(adjacent)
        # nodes = tf.reshape(nodes, [-1, 1, 8])
        # nodes = tf.strided_slice(nodes, [0], [100], [1])

        # return Record(10, 10, 1, record.label, adjacent)
