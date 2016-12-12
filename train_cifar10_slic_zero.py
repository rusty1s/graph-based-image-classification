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

    def node_mapping(superpixel):
        color = superpixel.mean
        center = superpixel.absolute_center

        return {
            'order': superpixel.order,
            'blue': color[0],
            'green': color[1],
            'red': color[2],
            'count': superpixel.count,
            'x': center[0],
            'y': center[1],
        }

    def edge_mapping(from_superpixel, to_superpixel):
        return {}

    def node_features(node_attributes):
        return [
            node_attributes['red'],
            node_attributes['green'],
            node_attributes['blue'],
            node_attributes['count'],
            node_attributes['x'],
            node_attributes['y'],
        ]

    batch = cifar10.get_train_batch(0)

    for i in range(0, len(batch['images'])):
        image = batch['images'][i]
        label = batch['labels'][i]

        rep = image_to_slic_zero(image, 100, compactness=1.0,
                                 max_iterations=10, sigma=0.0)

        superpixels = extract_superpixels(image, rep)

        graph = create_superpixel_graph(superpixels, node_mapping,
                                        edge_mapping)

        fields = receptive_fields(graph, order, 1, 100, 9, node_features, 6)

        print('{}/{}'.format(i+1, len(batch['images'])))

    pass

# Only run if the script is executed directly.
if __name__ == '__main__':
    main()
