import tensorflow as tf
import numpy as np

from data import inputs

from .inference import inference


BATCH_SIZE = 128

EVAL_DATA = True
EVAL_INTERVAL_SECS = 60 * 5
RUN_ONCE = False

SCALE_INPUTS = 1.0
DISTORT_INPUTS = True
ZERO_MEAN_INPUTS = True


def evaluate(dataset, network, checkpoint_dir, eval_dir, batch_size=BATCH_SIZE,
             scale_inputs=SCALE_INPUTS, distort_inputs=DISTORT_INPUTS,
             zero_mean_inputs=ZERO_MEAN_INPUTS, eval_data=EVAL_DATA,
             eval_interval_secs=EVAL_INTERVAL_SECS, run_once=RUN_ONCE):

    if not tf.gfile.Exists(checkpoint_dir):
        raise ValueError('Checkpoint directory {} doesn\'t exist.'
                         .format(checkpoint_dir))

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)

    with tf.Graph().as_default() as g:
        data, labels = inputs(dataset, batch_size, scale_inputs,
                              distort_inputs, zero_mean_inputs, num_epochs=1,
                              shuffle=False)

        logits = inference(data, network)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                path = ckpt.model_checkpoint_path

                # Restore from checkpoint.
                saver.restore(sess, path)

                # Assuming model_checkpoint_path looks something like
                # /my-favorite-path/model.ckpt-0, extract global step from it.
                global_step = path.split('/')[-1].split('-')[-1]

            else:
                print('No checkpoint file found.')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            true_count = 0

            if eval_data:
                total_count = dataset.num_examples_per_epoch_for_eval
            else:
                total_count = dataset.num_examples_per_epoch_for_train_eval

            step = 0

            try:
                while(True):
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    print(step)

            except tf.errors.OutOfRange:
                precision = true_count / total_count
                print('precision', precision)

                coord.request_stop()
                coord.join(threads)


def evaluate_from_config(dataset, config, eval_data=EVAL_DATA,
                         eval_interval_secs=EVAL_INTERVAL_SECS,
                         run_once=RUN_ONCE):

    evaluate(dataset,
             config['network'],
             config['checkpoint_dir'],
             config['eval_dir'],
             config.get('batch_size', BATCH_SIZE),
             config.get('scale_inputs', SCALE_INPUTS),
             config.get('distort_inputs', DISTORT_INPUTS),
             config.get('zero_mean_inputs', ZERO_MEAN_INPUTS),
             eval_data, eval_interval_secs, run_once)
