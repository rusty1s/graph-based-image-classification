import json

import tensorflow as tf

from data import inputs
from .inference import inference
from .model import train_step, cal_loss, cal_accuracy
from .hooks import hooks


CHECKPOINT_DIR = '/tmp/train'
BATCH_SIZE = 128
LAST_STEP = 20000

LEARNING_RATE = 0.1
EPSILON = 1.0
BETA_1 = 0.9
BETA_2 = 0.999

SCALE_INPUTS = 1.0
DISTORT_INPUTS = True
ZERO_MEAN_INPUTS = True

DISPLAY_STEP = 10
SAVE_CHECKPOINT_SECS = 30
SAVE_SUMMARIES_STEPS = 100


def train(dataset, network, checkpoint_dir=CHECKPOINT_DIR,
          batch_size=BATCH_SIZE, last_step=LAST_STEP,
          learning_rate=LEARNING_RATE, epsilon=EPSILON, beta1=BETA_1,
          beta2=BETA_2, scale_inputs=SCALE_INPUTS,
          distort_inputs=DISTORT_INPUTS, zero_mean_inputs=ZERO_MEAN_INPUTS,
          display_step=DISPLAY_STEP, save_checkpoint_secs=SAVE_CHECKPOINT_SECS,
          save_summaries_steps=SAVE_SUMMARIES_STEPS):

    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)
    tf.gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        data, labels = inputs(dataset, False, batch_size, scale_inputs,
                              distort_inputs, zero_mean_inputs, shuffle=True)

        logits = inference(data, network)
        loss = cal_loss(logits, labels)
        acc = cal_accuracy(logits, labels)

        train_op = train_step(
            loss, global_step, learning_rate, beta1, beta2, epsilon)

        try:
            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=checkpoint_dir,
                    save_checkpoint_secs=save_checkpoint_secs,
                    save_summaries_steps=save_summaries_steps,
                    hooks=hooks(display_step, last_step, batch_size, loss, acc)
                    ) as monitored_session:
                while not monitored_session.should_stop():
                    monitored_session.run(train_op)

        except KeyboardInterrupt:
            pass


def train_per_config(dataset, obj, display_step=DISPLAY_STEP,
                     save_checkpoint_secs=SAVE_CHECKPOINT_SECS,
                     save_summaries_steps=SAVE_SUMMARIES_STEPS):

    train(
        dataset,
        obj['network'],
        obj['checkpoint_dir'] if 'checkpoint_dir' in obj else CHECKPOINT_DIR,
        obj['batch_size'] if 'batch_size' in obj else BATCH_SIZE,
        obj['last_step'] if 'last_step' in obj else LAST_STEP,
        obj['learning_rate'] if 'learning_rate' in obj else LEARNING_RATE,
        obj['epsilon'] if 'epsilon' in obj else EPSILON,
        obj['beta1'] if 'beta1' in obj else BETA_1,
        obj['beta2'] if 'beta2' in obj else BETA_2,
        obj['scale_inputs'] if 'scale_inputs' in obj else SCALE_INPUTS,
        obj['distort_inputs'] if 'distort_inputs' in obj else DISTORT_INPUTS,
        obj['zero_mean_inputs'] if 'zero_mean_inputs' in obj
        else ZERO_MEAN_INPUTS,
        display_step, save_checkpoint_secs, save_summaries_steps)
