from __future__ import print_function

import time
from datetime import datetime

from .logger import LoggerHook


class TimeLoggerHook(LoggerHook):

    def __init__(self, display_step, batch_size, last_step):
        super().__init__(display_step)

        self._batch_size = batch_size
        self._last_step = last_step

    def before_display_step_run(self, run_context):
        self._start_time = time.time()

    def after_display_step_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        remaining = (self._last_step - self._step) * duration / 60
        examples_per_sec = self._batch_size / duration

        s = ', '.join([
            '[{:.1f} examples/sec'.format(examples_per_sec),
            '{:.2f} sec/batch'.format(duration),
            '{:.1f} min remaining]'.format(remaining),
            ])

        print(s, end=' ')
