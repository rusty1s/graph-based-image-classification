import os
import sys

from six.moves import xrange
import tensorflow as tf
from skimage.io import imsave

from data import datasets
from data import inputs


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar-10',
                           """The dataset to load.""")
tf.app.flags.DEFINE_string('data_dir', None,
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('save_images', False,
                            """Creates an images directory in the data
                            directory where images for training and
                            evaluation are saved in label directories
                            respectively.""")


def save_image_sets(dataset):
    """Saves images into an images folder in the datasets data directory.

    Args:
        dataset: The dataset.
    """

    images_dir = os.path.join(dataset.data_dir, 'images')

    # Abort if directory already exists.
    if tf.gfile.Exists(images_dir):
        return

    # Save the training images.
    train_dir = os.path.join(images_dir, 'train')
    save_images(dataset, train_dir, eval_data=False)

    # Save the evaluation images.
    eval_dir = os.path.join(images_dir, 'eval')
    save_images(dataset, eval_dir, eval_data=True)


def save_images(dataset, images_dir, eval_data):
    # Create a subfolder for every label.
    for label in dataset.labels:
        tf.gfile.MakeDirs(os.path.join(images_dir, label))

    image_names = {label: 0 for label in dataset.labels}

    image_batch, label_batch = inputs(dataset, distort_inputs=False,
                                      num_epochs=1, shuffle=False,
                                      eval_data=eval_data)

    image_batch = tf.cast(image_batch, tf.uint8)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if not eval_data:
        num_images = dataset.num_examples_per_epoch_for_train
    else:
        num_images = dataset.num_examples_per_epoch_for_eval

    try:
        count = 0

        while(True):
            images, labels = sess.run([image_batch, label_batch])

            for i in xrange(images.shape[0]):
                label_name = dataset.label_name(labels[i])
                image = images[i]

                image_names[label_name] += 1
                image_name = '{}.png'.format(image_names[label_name])
                image_path = os.path.join(images_dir, label_name, image_name)
                imsave(image_path, image)

                count += 1

                sys.stdout.write(
                    '\r>> Saving images to {} {:.1f}%'
                    .format(images_dir, 100.0 * count / num_images))
                sys.stdout.flush()

    except (tf.errors.OutOfRangeError, KeyboardInterrupt):
        pass

    finally:
        coord.request_stop()
        coord.join(threads)

        sess.close()

        print('')
        print('Successfully saved {} images to {}.'.format(count, images_dir))


def main(argv=None):
    """Runs the script."""

    if FLAGS.dataset not in datasets:
        raise ValueError('{} no valid dataset.'.format(FLAGS.dataset))

    if FLAGS.data_dir:
        dataset = datasets[FLAGS.dataset](FLAGS.data_dir)
    else:
        dataset = datasets[FLAGS.dataset]()

    if FLAGS.save_images:
        save_image_sets(dataset)

if __name__ == '__main__':
    tf.app.run()
