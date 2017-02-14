import sys

import tensorflow as tf
import numpy as np

from data import inputs

from .inference import inference
from .model import MOVING_AVERAGE_DECAY


BATCH_SIZE = 128

EVAL_DATA = True

SCALE_INPUTS = 1.0
DISTORT_INPUTS = True
ZERO_MEAN_INPUTS = True


def evaluate(dataset, network, checkpoint_dir, eval_dir, batch_size=BATCH_SIZE,
             scale_inputs=SCALE_INPUTS, distort_inputs=DISTORT_INPUTS,
             zero_mean_inputs=ZERO_MEAN_INPUTS, eval_data=EVAL_DATA):

    if not tf.gfile.Exists(checkpoint_dir):
        raise ValueError('Checkpoint directory {} doesn\'t exist.'
                         .format(checkpoint_dir))

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)

    with tf.Graph().as_default() as g:
        data, labels = inputs(dataset, eval_data, batch_size, scale_inputs,
                              distort_inputs, zero_mean_inputs, num_epochs=1,
                              shuffle=False)

        keep_prob = tf.placeholder(tf.float32)
        logits = inference(data, network, keep_prob)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        init_op = [tf.global_variables_initializer(),
                   tf.local_variables_initializer()]

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir, g)

        with tf.Session() as sess:
            sess.run(init_op)

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

            num_examples = 0

            try:
                while(True):
                    predictions = sess.run([top_k_op],
                                           feed_dict={keep_prob: 1.0})
                    true_count += np.sum(predictions)
                    num_examples += batch_size
                    num_examples = min(num_examples, total_count)

                    percentage = 100.0 * num_examples / total_count
                    sys.stdout.write('\r>> Calculating accuracy {:.1f}%'
                                     .format(percentage))
                    sys.stdout.flush()

            except (KeyboardInterrupt, tf.errors.OutOfRangeError):
                pass

            finally:
                precision = 100.0 * true_count / total_count

                print('')
                print('Accuracy: {:.2f}%'.format(precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision', simple_value=precision)
                summary_writer.add_summary(summary, global_step)

                coord.request_stop()
                coord.join(threads)


def evaluate_from_config(dataset, config, eval_data=EVAL_DATA):
    evaluate(dataset,
             config['network'],
             config['checkpoint_dir'],
             config['eval_dir'],
             config.get('batch_size', BATCH_SIZE),
             config.get('scale_inputs', SCALE_INPUTS),
             config.get('distort_inputs', DISTORT_INPUTS),
             config.get('zero_mean_inputs', ZERO_MEAN_INPUTS))
