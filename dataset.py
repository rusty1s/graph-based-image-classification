import os

import tensorflow as tf

from data import datasets


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar-10',
                           """The dataset to load.""")
tf.app.flags.DEFINE_string('data_dir', None,
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('save_images', False,
                            """Creates an images directory into the data
                            directory where images for training and
                            evaluation are saved in label directories
                            respectively.""")


def main(argv=None):
    """Runs the script."""

    if FLAGS.dataset not in datasets:
        raise ValueError('{} is no valid dataset.'.format(FLAGS.dataset))

    if FLAGS.data_dir:
        dataset = datasets[FLAGS.dataset](FLAGS.data_dir)
    else:
        dataset = datasets[FLAGS.dataset]()

    if FLAGS.save_images:
        dataset.save_images()

if __name__ == '__main__':
    tf.app.run()
