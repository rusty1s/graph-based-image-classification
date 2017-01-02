import json

import tensorflow as tf

from data import inputs
from .inference import inference
from .model import (train_step, cal_loss, cal_acc)
from .hooks import hooks


def train(dataset, train_dir, network_params_path):
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    with open(network_params_path, 'r') as f:
        network_params = json.load(f)

    structure = network_params['structure']
    batch_size = network_params['batch_size']
    last_step = network_params['last_step']

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        data, labels = inputs(dataset, batch_size)
        logits = inference(data, structure)
        loss = cal_loss(logits, labels)
        acc = cal_acc(logits, labels)
        train_op = train_step(loss, global_step, learning_rate=0.1,
                              epsilon=1.0)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_dir,
                save_checkpoint_secs=30,
                hooks=hooks(display_step=10, last_step=last_step,
                            batch_size=batch_size, loss=loss, accuracy=acc)
                ) as monitored_session:
            while not monitored_session.should_stop():
                monitored_session.run(train_op)
