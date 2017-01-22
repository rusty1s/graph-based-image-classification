import os
import sys
import warnings

import tensorflow as tf
import numpy as np
from skimage.io import imsave
from skimage.segmentation import mark_boundaries
from skimage.future import graph
from skimage.measure import regionprops
from skimage import draw


from data import datasets
from data import iterator
from superpixel.algorithm import generators


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar_10',
                           """The dataset. See dataset/__init__.py for a list
                           of all available datasets.""")
tf.app.flags.DEFINE_string('data_dir', None,
                           """Path to the data directory.""")
tf.app.flags.DEFINE_string('algorithm', 'slic',
                           """The superpixel algorithm. See
                           superpixel/algorithm/__init__.py for a list of all
                           available superpixel algorithms.""")
tf.app.flags.DEFINE_boolean('draw_graph', False,
                            """Draws an additional region adjacency graph.""")


def save_superpixel_images(dataset, algorithm, eval_data):
    """Saves images with computed superpixel boundaries for either training or
    evaluation to an images directory into the datasets data directory.

    Args:
        dataset: The dataset.
        algorithm: The superpixel algorithm.
        eval_data: Boolean indicating if one should use the train or eval data
          set.
    """

    dirname = 'eval' if eval_data else 'train'
    images_dir = os.path.join(dataset.data_dir, FLAGS.algorithm, dirname)

    # Abort if directory already exists.
    if tf.gfile.Exists(images_dir):
        return

    # Create a subdirectory for every label.
    for label in dataset.labels:
        tf.gfile.MakeDirs(os.path.join(images_dir, label))

    image_names = {label: 0 for label in dataset.labels}

    iterate = iterator(dataset, eval_data)

    def _before(image, label):
        segmentation = algorithm(image)

        # Cast image to uint8, so we can save it easily.
        return [tf.cast(image, tf.uint8), segmentation, label]

    def _each(output, index, last_index):
        # Get the image and the label name from the output of the session.
        image = output[0]
        segmentation = output[1]
        label_name = dataset.label_name(output[2])

        # Draw boundaries.
        output_image = mark_boundaries(image, segmentation, (0, 0, 0))

        # Draw region adjacency graph.
        if FLAGS.draw_graph:
            rag = graph.rag_mean_color(image, segmentation)

            # Offset is 1 so that regionprops does not ignore 0.
            offset = 1
            map_array = np.arange(segmentation.max() + 1)
            for n, d in rag.nodes_iter(data=True):
                for label in d['labels']:
                    map_array[label] = offset
                offset += 1

            rag_labels = map_array[segmentation]
            regions = regionprops(rag_labels)

            # Save the centroids in the node properties.
            for (n, data), region in zip(rag.nodes_iter(data=True), regions):
                data['centroid'] = region['centroid']

            # Iterate over all edges and draw them.
            for n1, n2, data in rag.edges_iter(data=True):
                y1, x1 = map(int, rag.node[n1]['centroid'])
                y2, x2 = map(int, rag.node[n2]['centroid'])
                line = draw.line(y1, x1, y2, x2)
                output_image[line] = [0, 1, 0]

                y1 = image.shape[0] - 2 if y1 >= image.shape[0] - 1 else y1
                x1 = image.shape[1] - 2 if x1 >= image.shape[1] - 1 else x1
                y1 = 1 if y1 < 1 else y1
                x1 = 1 if x1 < 1 else x1

                circle = draw.circle(y1, x1, 2)
                output_image[circle] = [1, 1, 0]

        # Save the image in the label named subdirectory and name it
        # incrementally.
        image_names[label_name] += 1
        image_name = '{}.png'.format(image_names[label_name])
        image_path = os.path.join(images_dir, label_name, image_name)

        # Disable precision lose warning when saving.
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
    iterate(_each, _before, _done)


def main(argv=None):
    """Runs the script."""

    if FLAGS.dataset not in datasets:
        raise ValueError('{} is no valid dataset.'.format(FLAGS.dataset))

    if FLAGS.algorithm not in generators:
        raise ValueError('{} is no valid superpixel algorithm.'
                         .format(FLAGS.algorithm))

    if FLAGS.data_dir:
        dataset = datasets[FLAGS.dataset](FLAGS.data_dir)
    else:
        dataset = datasets[FLAGS.dataset]()

    algorithm = generators[FLAGS.algorithm]()

    # Save images for training and evaluation.
    save_superpixel_images(dataset, algorithm, eval_data=False)
    save_superpixel_images(dataset, algorithm, eval_data=True)


if __name__ == '__main__':
    tf.app.run()
