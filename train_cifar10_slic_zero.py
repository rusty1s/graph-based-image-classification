from __future__ import print_function

import os

from cifar10 import Cifar10
from superpixels import (image_to_slic_zero, extract_superpixels,
                         create_superpixel_graph)
from graph import (receptive_fields, order)

# Possible arguments for the script. Look up each help parameter for additional
# information.
CIFAR10_DIR = os.path.join('.', 'datasets', 'cifar10')


# def get_arguments():
#   """Parses the arguments from the terminal and overrides their corresponding
#     default values. Returns all arguments in a dictionary."""

#     parser = argparse.ArgumentParser(description='Training script for the '
#                                      'CIFAR-10 train and test images')

#     parser.add_argument('-o', '--output', type=str, default=OUTPUT,
#                        help='Output directory in which to save the CIFAR-10 '
#                         'images. Default: {}'.format(OUTPUT))

#     return parser.parse_args()

def main():

    cifar10 = Cifar10(CIFAR10_DIR)
    cifar10.save_images()

    batch_1 = cifar10.get_train_batch(0)

    first_image = batch_1['images'][0]
    first_label = batch_1['labels'][0]

    rep = image_to_slic_zero(first_image, 100)
    superpixels = extract_superpixels(first_image, rep)

    def node_mapping(node):
        color = node.mean
        center = node.absolute_center

        return {
            'order': node.order,
            'blue': color[0],
            'green': color[1],
            'red': color[2],
            'count': node.count,
            'x': center[0],
            'y': center[1],
        }

    def edge_mapping(from_node, to_node):
        return {}

    graph = create_superpixel_graph(superpixels, node_mapping, edge_mapping)

    def node_features(attributes):
        return [
            attributes['red'],
            attributes['green'],
            attributes['blue'],
            attributes['count'],
            attributes['x'],
            attributes['y'],
        ]

    fields = receptive_fields(graph, order, 1, 100, 9, node_features, 6)

    pass

# Only run if the script is executed directly.
if __name__ == '__main__':
    main()
