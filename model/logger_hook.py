import time
from datetime import datetime

import tensorflow as tf


class LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, loss, batch_size, display_step=10):
        self._loss = loss
        self._batch_size = batch_size
        self._display_step = display_step

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()

        # Ask for loss value.
        return tf.train.SessionRunArgs(self._loss)

    def after_run(self, run_context, run_values):
        if self._step % self._display_step != 0:
            return

        loss_value = run_values.results
        duration = time.time() - self._start_time
        examples_per_sec = self._batch_size / duration
        datestring = '{:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())

        print('{}: step {}, loss = {:.2f} ({:.1f} examples/sec, {:.3f} '
              'sec/batch)'.format(datestring, self._step, loss_value,
                                  examples_per_sec, duration))
