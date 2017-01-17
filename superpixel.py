import os
import sys
import warnings

import tensorflow as tf
from skimage.io import imsave
from skimage.segmentation import mark_boundaries
from skimage.future import graph


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
                            """Draws an additional region adjacency graphs.""")


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

    iterate = iterator(dataset, eval_data=False, batch_size=1)

    def _before(image_batch, label_batch):
        # Remove the first dimension, because we only consider batch sizes of
        # one.
        image = tf.squeeze(image_batch, squeeze_dims=[0])
        label = tf.squeeze(label_batch, squeeze_dims=[0])
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
            output_image = graph.draw_rag(segmentation, rag, output_image)

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
