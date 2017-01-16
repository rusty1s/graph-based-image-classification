import numpy as np

from .superpixel import Superpixel


def extract(image, segmentation):
    """Returns an array of all the superpixels found in scan-line order of the
    superpixel representation with the superpixel values as identifiers."""

    height, width = segmentation.shape

    # Create the working dictionary of none initialized superpixels (except
    # their identifier and false bounding box coordinates).
    values = np.unique(segmentation)
    superpixels = {
        i: Superpixel(id=i, left=float('inf'), top=float('inf'))
        for i in values
    }

    bb = {
        i: {
            'right': float('-inf'),
            'bottom': float('-inf'),
        }
        for i in values
    }

    # Helper variable to calculate the scan-line order of the superpixels.
    order = 0

    # Iterate over each pixel once and collect as much information about
    # the individual superpixels as we can.
    for y, row in enumerate(segmentation):
        for x, value in enumerate(row):
            s = superpixels[value]

            # Set the order and increment if not already set for this
            # superpixel.
            if s.order is None:
                s.order = order
                order += 1

            # Compute the upperleft coordinates of the bounding box.
            s.left = min(s.left, x)
            s.top = min(s.top, y)

            current_bb = bb[value]
            current_bb['right'] = max(current_bb['right'], x)
            current_bb['bottom'] = max(current_bb['bottom'], y)

            # Calculate and update neighbors.
            slice_x = __get_1x1_slice(x, width)
            slice_y = __get_1x1_slice(y, height)
            slice_1x1 = segmentation[slice_y, slice_x]
            s.neighbors.update(slice_1x1.flatten())

    # Iterate over each superpixel and collect additional information.
    for s in superpixels.values():
        # Remove itself from neighborhood.
        s.neighbors.discard(s.id)

        # Slice the image to the bounding box.
        current_bb = bb[s.id]
        slice_x = slice(s.left, current_bb['right'] + 1)
        slice_y = slice(s.top, current_bb['bottom'] + 1)

        s.image = image[slice_y, slice_x]

        # Compute the mask.
        s.mask = __compute_mask(segmentation[slice_y, slice_x],
                                s.id)

    # Return the dictionary values as a list. Sort them ascending by the
    # `order` attribute.
    return sorted(list(superpixels.values()), key=lambda s: s.order)


def __get_1x1_slice(index, maximum):
    """Returns the 1x1 neighborhood of `index` exlcuding values lower than
    0 and greater than or equal to maximum."""

    return slice(max(0, index - 1), min(maximum, index + 2))


def __compute_mask(image, value):
    """Returns the mask of an image, where everything not equals `value`
    is 0 and everything equals `value` is 255."""

    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[image == value] = 255

    return mask
