import time
from datetime import datetime

import tensorflow as tf


class TimeLoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, loss, acc, batch_size, max_steps, display_step=10):
        self._loss = loss
        self._acc = acc
        self._batch_size = batch_size
        self._max_steps = max_steps
        self._display_step = display_step

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()

        # acc = tf.train.SessionRunArgs(self._acc)

        # Ask for loss value.
        return tf.train.SessionRunArgs([self._loss, self._acc])

    def after_run(self, run_context, run_values):
        if self._step % self._display_step != 0:
            return

        loss_value = run_values.results[0]
        acc_value = run_values.results[1]
        duration = time.time() - self._start_time
        examples_per_sec = self._batch_size / duration
        datestring = '{:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
        time_remaining = (self._max_steps - self._step) * duration / 60

        print('{}: step {}, loss = {:.2f}, accuracy = {:.4f} ({:.1f} '
              'examples/sec, {:.2f} sec/batch, {:.1f} min remaining till '
              'step {})'.format(datestring, self._step, loss_value, acc_value,
                                examples_per_sec, duration, time_remaining,
                                self._max_steps))
