import os
import sys
from six.moves import xrange

import tensorflow as tf
from skimage.io import imsave

from .inputs import inputs


def save_images(dataset, show_progress=True):
    """Saves images for training and evaluation to an images directory into the
    datasets data directory.

    Args:
        dataset: The dataset.
        show_progress: Show a pretty progress bar.
    """

    images_dir = os.path.join(dataset.data_dir, 'images')

    # Abort if directory already exists.
    if tf.gfile.Exists(images_dir):
        return

    train_image_dir = os.path.join(images_dir, 'train')
    _save_images(dataset, train_image_dir, False, show_progress)

    eval_image_dir = os.path.join(images_dir, 'eval')
    _save_images(dataset, eval_image_dir, True, show_progress)


def _save_images(dataset, images_dir, eval_data, show_progress):
    """Saves images of an image set into an directory.

    Args:
        dataset: The dataset.
        images_dir: The directory to save to.
        eval_data: Boolean whether to use the evaluation or the training image
         set.
        show_progress: Show a pretty progress bar.
    """

    # Create a subdirectory for every label.
    for label in dataset.labels:
        tf.gfile.MakeDirs(os.path.join(images_dir, label))

    image_names = {label: 0 for label in dataset.labels}

    # Read every image without distortions exactly once.
    image_batch, label_batch = inputs(dataset, distort_inputs=False,
                                      num_epochs=1, shuffle=False,
                                      eval_data=eval_data)

    # Cast to uint8 to easily write the image later on.
    image_batch = tf.cast(image_batch, tf.uint8)

    # Create a tf session to run the graph.
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

                # Save the image in the label named subdirectory and name it
                # incrementally.
                image_names[label_name] += 1
                image_name = '{}.png'.format(image_names[label_name])
                image_path = os.path.join(images_dir, label_name, image_name)

                imsave(image_path, image)

                count += 1

                if show_progress:
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

        if show_progress:
            print('')

        print('Successfully saved {} images to {}.'.format(count, images_dir))
