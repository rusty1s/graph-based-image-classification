import json

import tensorflow as tf

from data import datasets
from patchy import PatchySan


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None,
                           """Configuration of the dataset.""")


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

    with open(FLAGS.config, 'r') as f:
        config = json.load(f)

    dataset(config)


if __name__ == '__main__':
    tf.app.run()
