from __future__ import print_function

import tensorflow as tf

from .logger import LoggerHook


class LossLoggerHook(LoggerHook):

    def __init__(self, display_step, loss):
        super().__init__(display_step)

        self._loss = loss

    def before_display_step_run(self, run_context):
        return tf.train.SessionRunArgs(self._loss)

    def after_display_step_run(self, run_context, run_values):
        loss = run_values.results

        s = '[loss = {:.2f}]'.format(loss)

        print(s, end=' ')
