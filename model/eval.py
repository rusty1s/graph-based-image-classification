import tensorflow as tf


CHECKPOINT_DIR = '/tmp/train'
EVAL_DIR = '/tmp/eval'
BATCH_SIZE = 128

EVAL_DATA = False
EVAL_INTERVAL_SECS = 60 * 5
RUN_ONCE = False

SCALE_INPUTS = 1.0
DISTORT_INPUTS = True
ZERO_MEAN_INPUTS = True

DISPLAY_STEP = 10


def evaluate(dataset, checkpoint_dir=CHECKPOINT_DIR, eval_dir=EVAL_DIR,
             eval_data=EVAL_DATA, eval_interval_secs=EVAL_INTERVAL_SECS,
             run_once=RUN_ONCE):

    if not tf.gfile.Exists(checkpoint_dir):
        raise ValueError('Checkpoint directory {} doesn\'t exist.'
                         .format(checkpoint_dir))

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)


def json_evaluate(dataset, json, eval_data=EVAL_DATA,
                  eval_interval_secs=EVAL_INTERVAL_SECS, run_once=RUN_ONCE):

    evaluate(
        dataset,
        json['checkpoint_dir'] if 'checkpoint_dir' in json else CHECKPOINT_DIR,
        json['eval_dir'] if 'eval_dir' in json else EVAL_DIR,
        eval_data, eval_interval_secs, run_once)
