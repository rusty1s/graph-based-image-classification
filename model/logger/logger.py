import tensorflow as tf


class LoggerHook(tf.train.SessionRunHook):

    def __init__(self, display_step):
        self._display_step = display_step

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        self._step += 1

        if self._step % self._display_step == 0:
            return self.before_display_step_run(run_context)

    def after_run(self, run_context, run_values):
        if self._step % self._display_step == 0:
            self.after_display_step_run(run_context, run_values)

    def before_display_step_run(self, run_context):
        pass

    def after_display_step_run(self, run_context, run_values):
        pass
