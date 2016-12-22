from __future__ import print_function

from .logger import LoggerHook


class EolLoggerHook(LoggerHook):

    def __init__(self, display_step):
        super().__init__(display_step)

    def after_display_step_run(self, run_context, run_values):
        print('')
