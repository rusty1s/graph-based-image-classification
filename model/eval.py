import tensorflow as tf


BATCH_SIZE = 128

EVAL_DATA = True
EVAL_INTERVAL_SECS = 60 * 5
RUN_ONCE = False

SCALE_INPUTS = 1.0
DISTORT_INPUTS = True
ZERO_MEAN_INPUTS = True


def evaluate(dataset, checkpoint_dir, eval_dir=EVAL_DIR,
             scale_inputs=SCALE_INPUTS, distort_inputs=DISTORT_INPUTS,
             zero_mean_inputs=ZERO_MEAN_INPUTS, eval_data=EVAL_DATA,
             eval_interval_secs=EVAL_INTERVAL_SECS, run_once=RUN_ONCE):

    if not tf.gfile.Exists(checkpoint_dir):
        raise ValueError('Checkpoint directory {} doesn\'t exist.'
                         .format(checkpoint_dir))

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)


def evaluate_from_config(dataset, config, eval_data=EVAL_DATA,
                         eval_interval_secs=EVAL_INTERVAL_SECS,
                         run_once=RUN_ONCE):

    evaluate(dataset,
             config['checkpoint_dir'],
             config['eval_dir'],
             config.get('scale_inputs', SCALE_INPUTS),
             config.get('distort_inputs', DISTORT_INPUTS),
             config.get('zero_mean_inputs', ZERO_MEAN_INPUTS),
             eval_data, eval_interval_secs, run_once)
