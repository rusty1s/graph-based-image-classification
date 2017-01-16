import os
import sys

import tensorflow as tf
from skimage.io import imsave


from data import datasets
from data import iterator
from superpixel.algorithm import slic_generators
from superpixel import extract


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar_10',
                           """The dataset.""")
tf.app.flags.DEFINE_string('data_dir', None,
                           """Path to the data directory.""")
tf.app.flags.DEFINE_string('algorithm', 'slic',
                           """The slic superpixel algorithm.""")
tf.app.flags.DEFINE_integer('num_superpixels', 100,
                            """The number of superpixels.""")
tf.app.flags.DEFINE_float('compactness', 1.0,
                          """Balances color proximity and space proximity.
                          A higher value gives more weight to space proximity
                          making superpixel shapes more square/cubic.""")
tf.app.flags.DEFINE_integer('max_iterations', 10,
                            """Maximum number of iterations of k-means.""")
tf.app.flags.DEFINE_float('sigma', 0.0,
                          """Width of gaussian smoothing kernel for pre-
                          processing.""")


def to_superpixel_image(image, segmentation):
    return image


def save_superpixel_images(dataset, algorithm, eval_data):
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
        image = to_superpixel_image(image, segmentation)

        # Cast image to uint8, so we can save it easily.
        return [tf.cast(image, tf.uint8), label]

    def _each(output, index, last_index):
        # Get the image and the label name from the output of the session.
        image = output[0]
        label_name = dataset.label_name(output[1])

        # Save the image in the label named subdirectory and name it
        # incrementally.
        image_names[label_name] += 1
        image_name = '{}.png'.format(image_names[label_name])
        image_path = os.path.join(images_dir, label_name, image_name)

        imsave(image_path, image)

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

    if FLAGS.algorithm not in slic_generators:
        raise ValueError('{} is no valid slic superpixel algorithm.'
                         .format(FLAGS.algorithm))

    if FLAGS.data_dir:
        dataset = datasets[FLAGS.dataset](FLAGS.data_dir)
    else:
        dataset = datasets[FLAGS.dataset]()

    algorithm = slic_generators[FLAGS.algorithm](
        FLAGS.num_superpixels,
        FLAGS.compactness,
        FLAGS.max_iterations,
        FLAGS.sigma)

    # Save images for training and evaluation.
    save_superpixel_images(dataset, algorithm, eval_data=False)
    save_superpixel_images(dataset, algorithm, eval_data=True)


if __name__ == '__main__':
    tf.app.run()
