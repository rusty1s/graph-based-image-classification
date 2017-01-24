import os
import sys
import json

import tensorflow as tf

from data import DataSet, iterator, read_tfrecord, write_to_tfrecord
from patchy import labelings, neighborhood_assemblies
from .helper.node_sequence import node_sequence


DATA_DIR = '/tmp/patchy_san_data'
FORCE_WRITE = False
WRITE_NUM_EPOCHS = 1
DISTORT_INPUTS = False

NODE_LABELING = 'identity'
NUM_NODES = 100
NODE_STRIDE = 1
NEIGHBORHOOD_ASSEMBLY = 'by_weight'
NEIGHBORHOOD_SIZE = 7

INFO_FILENAME = 'info.json'
TRAIN_FILENAME = 'train.tfrecords'
TRAIN_INFO_FILENAME = 'train_info.json'
TRAIN_EVAL_FILENAME = 'train_eval.tfrecords'
TRAIN_EVAL_INFO_FILENAME = 'train_eval_info.json'
EVAL_FILENAME = 'eval.tfrecords'
EVAL_INFO_FILENAME = 'eval_info.json'


class PatchySan(DataSet):

    def __init__(self, dataset, grapher, data_dir=None,
                 force_write=FORCE_WRITE, write_num_epochs=WRITE_NUM_EPOCHS,
                 distort_inputs=DISTORT_INPUTS,
                 node_labeling=NODE_LABELING, num_nodes=NUM_NODES,
                 node_stride=NODE_STRIDE,
                 neighborhood_assembly=NEIGHBORHOOD_ASSEMBLY,
                 neighborhood_size=NEIGHBORHOOD_SIZE,
                 show_progress=None):

        data_dir = DATA_DIR if data_dir is None else data_dir
        self._dataset = dataset
        self._num_nodes
        self._neighborhood_size = neighborhood_size

        super().__init__(data_dir, show_progress)

        node_labeling = labelings[node_labeling]
        neighborhood_assembly = neighborhood_assemblies[neighborhood_assembly]

        tf.gfile.MakeDirs(DATA_DIR)

        train_file = os.path.join(DATA_DIR, TRAIN_FILENAME)
        train_info_file = os.path.join(DATA_DIR, TRAIN_INFO_FILENAME)

        if not tf.gfile.Exists(train_file) or force_write:
            _write(dataset, grapher, False, train_file, train_info_file,
                   write_num_epochs, distort_inputs, True, node_labeling,
                   num_nodes, node_stride, neighborhood_assembly,
                   neighborhood_size, self._show_progress)

        eval_file = os.path.join(DATA_DIR, EVAL_FILENAME)
        eval_info_file = os.path.join(DATA_DIR, EVAL_INFO_FILENAME)

        if not tf.gfile.Exists(eval_file) or force_write:
            _write(dataset, grapher, True, eval_file, eval_info_file,
                   1, distort_inputs, False, node_labeling, num_nodes,
                   node_stride, neighborhood_assembly, neighborhood_size,
                   self._show_progress)

        train_eval_file = os.path.join(DATA_DIR, TRAIN_EVAL_FILENAME)
        train_eval_info_file = os.path.join(DATA_DIR, TRAIN_EVAL_INFO_FILENAME)

        if distort_inputs and (not tf.gfile.Exists(train_eval_file) or
                               force_write):

            _write(dataset, grapher, False, train_eval_file,
                   train_eval_info_file, 1, True, False, node_labeling,
                   num_nodes, node_stride, neighborhood_assembly,
                   neighborhood_size, self._show_progress)

    @property
    def train_filenames(self):
        pass

    @property
    def eval_filenames(self):
        pass

    @property
    def train_eval_filenames(self):
        return self.train_filenames(self)

    @property
    def labels(self):
        return self._dataset.labels

    @property
    def num_examples_per_epoch_for_train(self):
        return self._dataset.num_examples_per_epoch_for_train

    @property
    def num_examples_per_epoch_for_eval(self):
        return self._dataset.num_examples_per_epoch_for_eval

    @property
    def num_examples_per_epoch_for_train_eval(self):
        return self._dataset.num_examples_per_epoch_for_train_eval

    def read(self, filename_queue):
        # TODO Convert to feature map
        return read_tfrecord(filename_queue,
                             [self._num_nodes, self._neighborhood_size, 83])


def _write(dataset, grapher, eval_data, tfrecord_file, info_file,
           write_num_epochs, distort_inputs, shuffle,
           node_labeling, num_nodes, node_stride, neighborhood_assembly,
           neighborhood_size, show_progress=True):

    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    iterate = iterator(dataset, eval_data, distort_inputs=distort_inputs,
                       num_epochs=write_num_epochs, shuffle=shuffle)

    def _before(image, label):
        nodes, adjacency = grapher(image)
        sequence = node_labeling(adjacency)
        sequence = node_sequence(sequence, num_nodes, node_stride)
        neighborhood = neighborhood_assembly(adjacency, sequence,
                                             neighborhood_size)

        return [nodes, neighborhood, label]

    def _each(output, index, last_index):
        write_to_tfrecord(writer,
                          {'nodes': output[0], 'neighborhood': output[1]},
                          output[2])

        if show_progress:
            sys.stdout.write(
                '\r>> Saving graphs to {} {:.1f}%'
                .format(tfrecord_file, 100.0 * index / last_index))
            sys.stdout.flush()

    def _done(index, last_index):
        if show_progress:
            print('')

        print('Successfully saved {} graphs to {}.'
              .format(index, tfrecord_file))

        with open(info_file, 'w') as f:
            json.dump({'count': index}, f)

    iterate(_each, _before, _done)
