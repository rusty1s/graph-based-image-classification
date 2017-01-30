import tensorflow as tf

from .record import Record


# The ratio to crop the image.
CROP_RATIO = 0.75


def distort_image_for_train(record):
    """Applies random distortions for training to the image of a record.

    Args:
        record: The record.

    Returns:
        A new record object after applying distortions.
    """

    image = record.data
    crop_shape = _crop_shape(record.shape)

    with tf.name_scope('distort_image_for_train', values=[image, crop_shape]):

        # Randomly crop a [height, width] section of the image.
        image = tf.random_crop(image, crop_shape)

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust the saturation and the contrast of the image.
        image = tf.cast(image, tf.uint8)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.0)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.0)
        image = tf.cast(image, tf.float32)

    return Record(image, crop_shape, record.label)


def distort_image_for_eval(record):
    """Applies distortions for evaluation to the image of a record.

    Args:
        record: The record.

    Returns:
        A new record object after applying distortions.
    """

    image = record.data
    crop_shape = _crop_shape(record.shape)

    with tf.name_scope('distort_image_for_eval', values=[image, crop_shape]):

        # Crop the central [new_height, new_width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(
            image, crop_shape[0], crop_shape[1])

    return Record(image, crop_shape, record.label)


def _crop_shape(shape):
    """Calculates a new, smaller shape after cropping.

    Args:
        shape: A shape.

    Returns:
        A shape.
    """

    return [int(CROP_RATIO * shape[0]), int(CROP_RATIO * shape[1]), shape[2]]
