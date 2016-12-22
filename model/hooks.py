import tensorflow as tf

from .logger import (TimeLoggerHook,
                     LossLoggerHook,
                     AccuracyLoggerHook,
                     EolLoggerHook)


def hooks(display_step, last_step, batch_size, loss, accuracy):
    return [
        tf.train.StopAtStepHook(last_step=last_step),
        tf.train.NanTensorHook(loss),
        TimeLoggerHook(display_step, batch_size, last_step),
        LossLoggerHook(display_step, loss),
        AccuracyLoggerHook(display_step, accuracy),
        EolLoggerHook(display_step),
    ]
