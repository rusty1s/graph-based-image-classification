from __future__ import print_function

import tensorflow as tf

from .logger import LoggerHook


class MemoryLoggerHook(LoggerHook):

    def __init__(self, display_step):
        super().__init__(display_step)

    def after_display_step_run(self, run_context, run_values):
        memory = (self.process.get_memory_info()[0] / float(2 ** 20))

        s = '[memory usage = {} MB]'.format(memory)

        print(s, end=' ')
