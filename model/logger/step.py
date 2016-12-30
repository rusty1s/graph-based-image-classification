from __future__ import print_function

from .logger import LoggerHook


class StepLoggerHook(LoggerHook):

    def __init__(self, display_step, last_step):
        super().__init__(display_step)

        self._last_step_length = len(str(last_step))

    def after_display_step_run(self, run_context, run_values):
        s = '[step {:0{}d}]'.format(self._step, self._last_step_length)

        print(s, end=' ')
