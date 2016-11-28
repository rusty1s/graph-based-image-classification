""" Saves and loads a tensorflow session.
"""

from __future__ import print_function

import sys
import os


def save_checkpoint(saver, sess, checkpoint_dir, name, version):
    """Saves the session in the checkpoint_dir as {name}-{version}.ckpt."""

    print('Storing checkpoint to {}...'.format(checkpoint_dir), end=' ')
    sys.stdout.flush()  # don't wait for buffer before writing to terminal

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_name = '{}.ckpt'.format(name)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    # save checkpoint with version suffix
    saver.save(sess, checkpoint_path, global_step=version)

    print('Done.')


def load_latest_checkpoint(saver, sess, checkpoint_dir):
    """Loads the latest checkpoint in the checkpoint_dir into session."""

    print('Trying to restore saved '
          'checkpoints from {}...'.format(checkpoint_dir))
    sys.stdout.flush()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        path = ckpt.model_checkpoint_path

        print('Checkpoint found: {}'.format(path))

        # path is formatted as **/{name}-{version}
        name = path.split('/')[-1]
        version = name.split('-')[-1]

        print('Newest version is {}, restoring...'.format(version), end=' ')
        sys.stdout.flush()

        saver.restore(sess, path)

        print('Done.')
        return int(version)
    else:
        print('No checkpoint found.')
        return None
