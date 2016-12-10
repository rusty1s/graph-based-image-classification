import os
import cv2
import imutils
import numpy as np

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle


def save_superpixel_image(image, superpixels, path, name,
                          show_contour=True, contour_color=(0, 0, 0),
                          contour_thickness=1, show_center=False,
                          center_radius=2, center_color=(0, 0, 0),
                          show_mean=False):
    """Saves the superpixel representation of an image to a file. Visualizes
    the superpixels by drawing contours, the superpixels with its mean color
    and/or their center of mass."""

    output_image = np.array(image)

    for s in superpixels:
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
            # Draw a filled circle with the contour color.
            cv2.circle(output_image, s.rounded_absolute_center, center_radius,
                       center_color, -1)

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
