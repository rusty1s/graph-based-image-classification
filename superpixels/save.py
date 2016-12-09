import os
import cv2
import imutils
import numpy as np

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle

from .extract import extract_superpixels


def save_superpixel_image(image, superpixel_representation, path, name,
                          show_contour=True, contour_color=(0, 0, 0),
                          contour_thickness=1, show_center=True,
                          center_radius=2, show_mean=True):

    output_image = np.array(image)
    superpixels = extract_superpixels(image, superpixel_representation)

    for s in superpixels.values():
        if show_mean or show_contour:
            # For the drawing of the mean color and the contour, we need to
            # calculate the contour of the mask.
            #
            # **ATTENTION:** `findContours` modifies the source image, so we
            # need to copy the mask before passing it to OpenCV.
            contours = cv2.findContours(np.copy(s.mask), cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_NONE)

            # Grab the tuples based on whether we are using OpenCV 2.4 or 3.
            contours = contours[0] if imutils.is_cv2() else contours[1]

        if show_mean:
            # Draw the filled contour with the superpixels mean color.
            cv2.drawContours(output_image, contours, -1, s.mean, -1,
                             offset=(s.left, s.top))

        if show_contour:
            # Draw the outline of the contour with the contour color.
            cv2.drawContours(output_image, contours, -1, contour_color,
                             contour_thickness, offset=(s.left, s.top))

        if show_center:
            # We need to pass a center of integers.
            center = (int(s.absolute_center[0]), int(s.absolute_center[1]))

            # Draw a filled circle with the contour color.
            cv2.circle(output_image, center, center_radius, contour_color, -1)

    # Write the image to the specified output path.
    try:
        os.makedirs(path)
    except:
        pass

    cv2.imwrite(os.path.join(path, name), output_image)


def save_superpixels(superpixels, path, name):
    """Writes the generated dictionary of superpixels to disk."""

    try:
        os.makedirs(path)
    except:
        pass

    with open(os.path.join(path, name), 'wb') as output:
        pickle.dump(superpixels, output, -1)


def read_superpixels(path):
    """Reads the written dictionary of superpixels from disk."""

    with open(path, 'rb') as input:
        return pickle.load(input)
