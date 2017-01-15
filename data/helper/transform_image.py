import numpy as np
from skimage.transform import resize


def crop_shape_from_box(image, shape, box):
    """Crops a specified shape from an image which includes the maximal cropped
    image of the box. Performs rescaling if the the box is larger than the
    shape.

    Args:
        image: A numpy array.
        shape: A [height, width] shape.
        box: A [top, right, bottom, left] defined box.

    Returns:
        A cropped image.
    """

    bb_top = box[0]
    bb_height = box[2] - bb_top
    bb_left = box[3]
    bb_width = box[1] - bb_left
    bb_center_y = bb_top + bb_height // 2
    bb_center_x = bb_left + bb_width // 2

    if shape[0] < bb_height or shape[1] < bb_width:
        # We have a bounding box that is on at least one side greater than the
        # requested image shape. We need to crop the image, so that the
        # complete bounding box of the image is visible after cropping.

        # Find the side with the greater ratio between bounding size and shape
        # size.
        ratio_y = bb_height / shape[0]
        ratio_x = bb_width / shape[1]

        if ratio_y < ratio_x:
            width = bb_width
            height = min(image.shape[0], int(ratio_x * shape[0]))
        else:
            height = bb_height
            width = min(image.shape[1], int(ratio_y * shape[1]))

    else:
        # The bounding box is smaller than the requested shape. We try to crop
        # the bounding box from its center to the requested shape.
        height = shape[0]
        width = shape[1]

    # Crop the image based on the center of the bounding box.
    crop_top = max(bb_center_y - height // 2, 0)
    crop_left = max(bb_center_x - width // 2, 0)

    # We need to adjust the variables if the object is too far at the right
    # or the bottom of the image, so that we can get the maximal cropping
    # defined by the height and width.
    crop_top = min(crop_top, max(image.shape[0] - height, 0))
    crop_left = min(crop_left, max(image.shape[1] - width, 0))

    # Calculate the opposite sides of the cropping in case the image is
    # smaller than the defined cropping.
    crop_bottom = min(crop_top + height, image.shape[0])
    crop_right = min(crop_left + width, image.shape[1])

    # Finally crop the image.
    image = image[crop_top:crop_bottom, crop_left:crop_right]

    # Rescale the image if needed.
    return rescale_and_crop(image, shape)


def rescale_and_crop(image, shape):
    """Rescales an image to the specified shape. Performs cropping if the image
    dimensions do not match the specified shape.

    Args:
        image: A numpy array.
        shape: A [height, width] shape.

    Returns:
        A rescaled and cropped image.
    """

    # The passed image can be greater or smaller than the requested shape.
    # We need to either scale the image up or down and crop it if this is the
    # case.
    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image

    scale = max(1.0 * shape[0] / image.shape[0],
                1.0 * shape[1] / image.shape[1])

    # Calculate the shape after resizing and resize the image based on this
    # shape.
    post_shape = [max(int(scale * image.shape[0]), shape[0]),
                  max(int(scale * image.shape[1]), shape[1])]

    image = resize(image, post_shape, preserve_range=True)
    image = image.astype(np.uint8)

    # Finally crop the image again based on its center.
    crop_top = (image.shape[0] - shape[0]) // 2
    crop_bottom = crop_top + shape[0]
    crop_left = (image.shape[1] - shape[1]) // 2
    crop_right = crop_left + shape[1]

    return image[crop_top:crop_bottom, crop_left:crop_right]
