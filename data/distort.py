import tensorflow as tf

from .record import Record


def distort_image_for_train(record, new_height, new_width):
    """Applies random distortions for training to the image of a CIFAR-10
    record.

    Args:
        record: The record before applying distortions.
        new_height: The new height of the image.
        new_width: The new width of the image.

    Returns:
        A new record object of the passed record after applying distortions.
    """

    image = record.data

    with tf.name_scope('distort_for_train', values=[image]):

        # Randomly crop a [height, width] section of the image.
        image = tf.random_crop(image, [new_height, new_width, record.depth])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust the brightness of the image.
        image = tf.image.random_brightness(image, max_delta=63)

        # Ramdomly adjust the contrast of the image.
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        image = tf.image.per_image_standardization(image)

    return Record(new_height, new_width, record.depth, record.label, image)


def distort_image_for_eval(record, new_height, new_width):
    """Applies distortions for evaluation to the image of a record.

    Args:
        record: The record before applying distortions.
        new_height: The new height of the image.
        new_width: The new width of the image.

    Returns:
        A new record object of the passed record after applying distortions.
    """

    image = record.data

    with tf.name_scope('distort_image_for_eval', values=[image]):

        # Crop the central [new_height, new_width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(
            image, new_height, new_width)

        image = tf.image.per_image_standardization(image)

    return Record(new_height, new_width, record.depth, record.label, image)
