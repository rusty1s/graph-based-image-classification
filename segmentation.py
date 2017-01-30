import os
import sys
import warnings

import tensorflow as tf
import numpy as np
import networkx as nx
from skimage.io import imsave
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage import draw


from data import datasets, iterator
from segmentation.algorithm import generators as segmentations
from segmentation import adjacencies
from patchy import neighborhood_assemblies as neighborhoods


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'pascal_voc',
                           """The dataset. See dataset/__init__.py for a list
                           of all available datasets.""")
tf.app.flags.DEFINE_string('data_dir', None,
                           """Path to the data directory.""")
tf.app.flags.DEFINE_string('segmentation_algorithm', 'slic',
                           """The segmentation algorithm. See
                           segmentation/algorithm/__init__.py for a list of all
                           available segmentation algorithms.""")
tf.app.flags.DEFINE_string('adjacency_algorithm', 'euclidean_distance',
                           """The adjacency algorithm. See
                           segmentation/__init__.py for a list of all
                           available adjacency algorithms.""")
tf.app.flags.DEFINE_string('neighborhood_assembly', 'grid_spiral',
                           """The neighborhood assembly algorithm. See
                           patchy/helper/neighborhood_assembly.py for a list of
                           all available neighborhood assembly algorithms.""")


def draw_image(image, segmentation, adjacency, neighborhood):
    image = mark_boundaries(image, segmentation, (0, 0, 0))

    graph = nx.from_numpy_matrix(adjacency)

    segmentation += np.ones_like(segmentation)
    segments = regionprops(segmentation)

    # Save the centroids in the node properties.
    for (n, data), segment in zip(graph.nodes_iter(data=True), segments):
        data['centroid'] = segment['centroid']

    # Iterate over all edges and draw them.
    for n1, n2, data in graph.edges_iter(data=True):
        y1, x1 = map(int, graph.node[n1]['centroid'])
        y2, x2 = map(int, graph.node[n2]['centroid'])
        line = draw.line(y1, x1, y2, x2)

        if n1 in neighborhood and n2 in neighborhood:
            image[line] = [1, 0, 0]
        else:
            image[line] = [0, 1, 0]

    # Draw a circle at the root node.
    for i in range(0, neighborhood.shape[0]):
        if neighborhood[i] < 0:
            continue

        y1, x1 = graph.node[neighborhood[i]]['centroid']
        circle = draw.circle(y1, x1, 2)
        j = i/(neighborhood.shape[0] - 1)
        image[circle] = [j, j, j]

    return image


def iterate(dataset, segmentation_algorithm, adjacency_algorithm, eval_data,
            neighborhood_assembly, node, size):
    """Saves images with computed segment boundaries for either training or
    evaluation to an images directory into the datasets data directory.

    Args:
        dataset: The dataset.
        algorithm: The segmentation algorithm.
        eval_data: Boolean indicating if one should use the train or eval data
          set.
    """

    dirname = 'eval' if eval_data else 'train'
    images_dir = os.path.join(dataset.data_dir, FLAGS.segmentation_algorithm,
                              dirname)

    # Abort if directory already exists.
    if tf.gfile.Exists(images_dir):
        return

    # Create a subdirectory for every label.
    for label in dataset.labels:
        tf.gfile.MakeDirs(os.path.join(images_dir, label))

    image_names = {label: 0 for label in dataset.labels}

    _iterate = iterator(dataset, eval_data)

    def _before(image, label):
        segmentation = segmentation_algorithm(image)
        adjacency = adjacency_algorithm(segmentation)

        # Collect one neighborhood from one node.
        neighborhoods = neighborhood_assembly(adjacency, [node], size)
        neighborhood = tf.reshape(neighborhoods, [-1])

        return [tf.cast(image, tf.uint8), segmentation, adjacency,
                neighborhood, label]

    def _each(output, index, last_index):
        # Get the image and the label name from the output of the session.
        image = output[0]
        segmentation = output[1]
        adjacency = output[2]
        neighborhood = output[3]
        label_name = dataset.label_name(output[4])

        output_image = draw_image(image, segmentation, adjacency, neighborhood)

        # Save the image in the label named subdirectory and name it
        # incrementally.
        image_names[label_name] += 1
        image_name = '{}.png'.format(image_names[label_name])
        image_path = os.path.join(images_dir, label_name, image_name)

        # Disable precision loss warning when saving.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imsave(image_path, output_image)

        sys.stdout.write(
            '\r>> Saving images to {} {:.1f}%'
            .format(images_dir, 100.0 * index / last_index))
        sys.stdout.flush()

    def _done(index, last_index):
        print('')
        print('Successfully saved {} images to {}.'.format(index, images_dir))

    # Run through each single batch.
    _iterate(_each, _before, _done)


def main(argv=None):
    """Runs the script."""

    if FLAGS.data_dir:
        dataset = datasets[FLAGS.dataset](FLAGS.data_dir)
    else:
        dataset = datasets[FLAGS.dataset]()

    segmentation_algorithm = segmentations[FLAGS.segmentation_algorithm]()
    adjacency_algorithm = adjacencies[FLAGS.adjacency_algorithm]
    neighborhood_assembly = neighborhoods[FLAGS.neighborhood_assembly]

    # Save images for training and evaluation.
    iterate(dataset, segmentation_algorithm, adjacency_algorithm, False,
            neighborhood_assembly, node=100, size=18)
    iterate(dataset, segmentation_algorithm, adjacency_algorithm, True,
            neighborhood_assembly, node=100, size=18)


if __name__ == '__main__':
    tf.app.run()
