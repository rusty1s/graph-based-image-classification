from __future__ import print_function

import tensorflow as tf

from .logger import LoggerHook


class AccuracyLoggerHook(LoggerHook):

    def __init__(self, display_step, accuracy):
        super().__init__(display_step)

        self._accuracy = accuracy

    def before_display_step_run(self, run_context):
        return tf.train.SessionRunArgs(self._accuracy)

    def after_display_step_run(self, run_context, run_values):
        accuracy = run_values.results

        s = '[accuracy = {:.2f}] '.format(accuracy)

        print(s, end=' ')
