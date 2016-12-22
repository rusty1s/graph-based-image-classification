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
        examples_per_sec = self._batch_size / duration
        datestring = '{:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
        remaining = (self._last_step - self._step) * duration / 60

        s = ('[{}: step {}, {:.1f} min remaining, {:.1f} examples/sec, {:.2f} '
             'sec/batch]').format(datestring, self._step, remaining,
                                  examples_per_sec, duration)

        print(s, end=' ')
