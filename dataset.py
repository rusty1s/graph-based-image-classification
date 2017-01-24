import os
import sys

import tensorflow as tf
from skimage.io import imsave

from data import datasets
from data import iterator


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar_10',
                           """The dataset. See dataset/__init__.py for a list
                           of all available datasets.""")
tf.app.flags.DEFINE_string('data_dir', None,
                           """Path to the data directory.""")


def save_images(dataset, eval_data):
    """Saves images for either training or evaluation to an images directory
    into the datasets data directory.

    Args:
        dataset: The dataset.
        eval_data: Boolean indicating if one should use the train or eval data
          set.
    """

    dirname = 'eval' if eval_data else 'train'
    images_dir = os.path.join(dataset.data_dir, 'images', dirname)

    # Abort if directory already exists.
    if tf.gfile.Exists(images_dir):
        return

    # Create a subdirectory for every label.
    for label in dataset.labels:
        tf.gfile.MakeDirs(os.path.join(images_dir, label))

    image_names = {label: 0 for label in dataset.labels}

    iterate = iterator(dataset, eval_data)

    def _before(image, label):
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

    if FLAGS.data_dir:
        dataset = datasets[FLAGS.dataset](FLAGS.data_dir)
    else:
        dataset = datasets[FLAGS.dataset]()

    # Save images for training and evaluation.
    save_images(dataset, eval_data=False)
    save_images(dataset, eval_data=True)

if __name__ == '__main__':
    tf.app.run()
