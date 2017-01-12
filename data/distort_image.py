import tensorflow as tf

from .record import Record


# The ratio to crop the image.
CROP_RATIO = 0.75


def distort_image_for_train(record):
    """Applies random distortions for training to the image of a record.

    Args:
        record: The record before applying distortions.

    Returns:
        A new record object of the passed record after applying distortions.
    """

    image = record.data
    crop_shape = _crop_shape(record.shape)

    with tf.name_scope('distort_image_for_train', values=[image, crop_shape]):

        # Randomly crop a [height, width] section of the image.
        image = tf.random_crop(image, crop_shape)

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust the brightness of the image.
        image = tf.image.random_brightness(image, max_delta=63)

        # Ramdomly adjust the contrast of the image.
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        image = tf.image.per_image_standardization(image)

    return Record(image, crop_shape, record.label)


def distort_image_for_eval(record, new_height, new_width):
    """Applies distortions for evaluation to the image of a record.

    Args:
        record: The record before applying distortions.

    Returns:
        A new record object of the passed record after applying distortions.
    """

    image = record.data
    crop_shape = _crop_shape(record.shape)

    with tf.name_scope('distort_image_for_eval', values=[image, crop_shape]):

        # Crop the central [new_height, new_width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(image, crop_shape)

        image = tf.image.per_image_standardization(image)

    return Record(image, crop_shape, record.label)


def _crop_shape(shape):
    return [int(CROP_RATIO * shape[0]), int(CROP_RATIO * shape[1]), shape[2]]
