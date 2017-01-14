from skimage.transform import resize


def crop_shape_from_box(image, shape, box):
    bb_top = box[0]
    bb_height = bb_top - box[2]
    bb_left = box[3]
    bb_width = bb_left - box[1]

    # Crop the bounding box or the maximal defined shape of the image,
    # whichever resolution is greater, so that we always crop the full
    # bounding box of the object into the image.
    height = max(bb_height, shape[0])
    width = max(bb_width, shape[1])

    # Crop the image based on the center of the bounding box.
    crop_top = max(bb_top + (bb_height - height) // 2, 0)
    crop_left = max(bb_left + (bb_width - width) // 2, 0)

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
    # The passed image can be greater or smaller than the wished fixed shape.
    # We need to either scale the image up or down and crop it again if this is
    # the case.
    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image

    scale = max(1.0 * shape[0] / image.shape[0],
                1.0 * shape[1] / image.shape[1])

    # Calculate the shape after resizing and resize the image based on this
    # shape.
    post_shape = [max(int(scale * image.shape[0]), shape[0]),
                  max(int(scale * image.shape[1]), shape[1])]
    image = resize(image, post_shape, preserve_range=True)

    # Finally crop the image again based on its center.
    crop_top = (image.shape[0] - shape[0]) // 2
    crop_bottom = crop_top + shape[0]
    crop_left = (image.shape[1] - shape[1]) // 2
    crop_right = crop_left + shape[1]

    return image[crop_top:crop_bottom, crop_left:crop_right]
