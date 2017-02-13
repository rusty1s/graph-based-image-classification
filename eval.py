import json

import tensorflow as tf

from data import datasets
from patchy import PatchySan
from model import evaluate_from_config


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None,
                           """Path to the configuration json file of the
                           network.""")
tf.app.flags.DEFINE_boolean('eval_data', True,
                            """Whether to use the eval data or not to eval.""")


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

    evaluate_from_config(dataset(config['dataset']), config, FLAGS.eval_data)


if __name__ == '__main__':
    tf.app.run()
