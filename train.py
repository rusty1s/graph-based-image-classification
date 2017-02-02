import json

import tensorflow as tf

from data import datasets
from patchy import PatchySan
from model import train_from_config


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None,
                           """Path to the configuration json file of the
                           network.""")
tf.app.flags.DEFINE_integer('display_step', 10,
                            """The frequency, in number of global steps, that
                            the network training is logged.""")
tf.app.flags.DEFINE_integer('save_checkpoint_secs', 60*60,
                            """The frequency, in seconds, that a checkpoint is
                            saved.""")
tf.app.flags.DEFINE_integer('save_summaries_steps', 100,
                            """The frequency, in number of global steps, that
                            the summaries are written to disk.""")


def dataset(config):
    """Reads and initializes a dataset specified by a passed configuration.

    Args:
        config: Configuration object.

    Returns:
        A dataset.
    """

    if config['name'] in datasets:
        return datasets[config['name']].create(config)
    elif config['name'] == 'patchy_san':
        return PatchySan.create(config)
    else:
        raise ValueError('Dataset not found.')


def main(argv=None):
    """Runs the script."""

    if not tf.gfile.Exists(FLAGS.config):
        raise ValueError('{} does not exist.'.format(FLAGS.config))

    with open(FLAGS.config, 'r') as f:
        config = json.load(f)

    train_from_config(dataset(config['dataset']), config,
                      FLAGS.display_step, FLAGS.save_checkpoint_secs,
                      FLAGS.save_summaries_steps)


if __name__ == '__main__':
    tf.app.run()
