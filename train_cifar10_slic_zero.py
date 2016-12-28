from __future__ import print_function

import os
from datetime import datetime
import numpy as np

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle

from cifar10 import Cifar10
from superpixels import (image_to_slic_zero, extract_superpixels,
                         create_superpixel_graph)
from graph import (receptive_fields, order)

# Possible arguments for the script. Look up each help parameter for additional
# information.
CIFAR10_DIR = os.path.join('.', 'datasets', 'cifar10')
DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

NUM_SEGMENTS = 100
COMPACTNESS = 1.0
MAX_ITERATIONS = 10
SIGMA = 0.0

STRIDE = 1
WIDTH = 100
SIZE = 9
NODE_FEATURE_SIZE = 6
NODE_FEATURE_1 = 'red'
NODE_FEATURE_2 = 'green'
NODE_FEATURE_3 = 'blue'
NODE_FEATURE_4 = 'count'
NODE_FEATURE_5 = 'x'
NODE_FEATURE_6 = 'y'


def main():

    cifar10 = Cifar10(CIFAR10_DIR)

    path = os.path.join(CIFAR10_DIR, 'slic_zero', DATESTRING)

    try:
        os.makedirs(path)
    except:
        pass

    with open(os.path.join(path, 'info.txt'), 'w') as file:
        file.write('NUM_SEGMENTS = {}\n'.format(NUM_SEGMENTS))
        file.write('COMPACTNESS = {}\n'.format(COMPACTNESS))
        file.write('MAX_ITERATIONS = {}\n'.format(MAX_ITERATIONS))
        file.write('SIGMA = {}\n'.format(SIGMA))
        file.write('STRIDE = {}\n'.format(STRIDE))
        file.write('WIDTH = {}\n'.format(WIDTH))
        file.write('SIZE = {}\n'.format(SIZE))
        file.write('NODE_FEATURE_SIZE = {}\n'.format(NODE_FEATURE_SIZE))
        file.write('NODE_FEATURE_1 = {}\n'.format(NODE_FEATURE_1))
        file.write('NODE_FEATURE_2 = {}\n'.format(NODE_FEATURE_2))
        file.write('NODE_FEATURE_3 = {}\n'.format(NODE_FEATURE_3))
        file.write('NODE_FEATURE_4 = {}\n'.format(NODE_FEATURE_4))
        file.write('NODE_FEATURE_5 = {}\n'.format(NODE_FEATURE_5))
        file.write('NODE_FEATURE_6 = {}\n'.format(NODE_FEATURE_6))

    def node_mapping(superpixel):
        return superpixel.get_attributes()

    def edge_mapping(from_superpixel, to_superpixel):
        return {}

    def node_features(node_attributes):
        return [
            node_attributes[NODE_FEATURE_1],
            node_attributes[NODE_FEATURE_2],
            node_attributes[NODE_FEATURE_3],
            node_attributes[NODE_FEATURE_4],
            node_attributes[NODE_FEATURE_5],
            node_attributes[NODE_FEATURE_6],
        ]

    for i in range(0, 5):
        batch = cifar10.get_train_batch(i)

        np_array = np.zeros(shape=(len(batch['data']), 1, WIDTH * SIZE,
                                   NODE_FEATURE_SIZE))

        print('Batch {}:'.format(i+1))
        print('========')

        for j in range(0, len(batch['data'])):
            image = batch['data'][j]
            # label = batch['labels'][j]

            rep = image_to_slic_zero(image, NUM_SEGMENTS,
                                     compactness=COMPACTNESS,
                                     max_iterations=MAX_ITERATIONS,
                                     sigma=SIGMA)

            superpixels = extract_superpixels(image, rep)

            graph = create_superpixel_graph(superpixels, node_mapping,
                                            edge_mapping)

            fields = receptive_fields(graph, order, STRIDE, WIDTH, SIZE,
                                      node_features, NODE_FEATURE_SIZE)

            np_array[j] = fields

            if (j+1) % 100 == 0:
                print('{}/{}'.format(j+1, len(batch['data'])))

        with open(os.path.join(path, 'data_batch_{}'.format(i+1)),
                  'wb') as output:

            pickle.dump({'data': np_array, 'labels': batch['labels']},
                        output, -1)

    batch = cifar10.get_test_batch()

    np_array = np.zeros(shape=(len(batch['data']), 1, WIDTH * SIZE,
                               NODE_FEATURE_SIZE))

    print('Test Batch:')
    print('==========')

    for j in range(0, len(batch['data'])):
        image = batch['data'][j]

        rep = image_to_slic_zero(image, NUM_SEGMENTS,
                                 compactness=COMPACTNESS,
                                 max_iterations=MAX_ITERATIONS,
                                 sigma=SIGMA)

        superpixels = extract_superpixels(image, rep)

        graph = create_superpixel_graph(superpixels, node_mapping,
                                        edge_mapping)

        fields = receptive_fields(graph, order, STRIDE, WIDTH, SIZE,
                                  node_features, NODE_FEATURE_SIZE)

        np_array[j] = fields

        if (j+1) % 100 == 0:
            print('{}/{}'.format(j+1, len(batch['data'])))

    with open(os.path.join(path, 'test_batch'), 'wb') as output:

        pickle.dump({'data': np_array, 'labels': batch['labels']}, output, -1)

# Only run if the script is executed directly.
if __name__ == '__main__':
    main()
