"""SLIC-zero Superpixel segmentation script
"""

from __future__ import print_function

import argparse
import cv2

from superpixels import (image_to_slic_zero, extract_superpixels,
                         save_superpixel_image)


# Possible arguments for the script. Look up each help parameter for additional
# information.
NUM_SEGMENTS = 100
COMPACTNESS = 1.0
MAX_ITERATIONS = 10
SIGMA = 0.0
SHOW_CONTOUR = True
CONTOUR_THICKNESS = 1
SHOW_CENTER = False
CENTER_RADIUS = 2
SHOW_MEAN = False


def get_arguments():
    """Parses the arguments from the terminal and overrides their corresponding
    default values. Returns all arguments in a dictionary."""

    parser = argparse.ArgumentParser(description='SLIC-zero Superpixel '
                                     'segmentation script')

    parser.add_argument('-i', '--image', required=True, type=str,
                        help='Path to the image')
    parser.add_argument('-o', '--output', type=str,
                        help='Path and name to the superpixel segmented image.'
                        ' Default: <image_path>/<image_name>-SLIC_zero.<ext>')

    parser.add_argument('--num-segments', type=int, default=NUM_SEGMENTS,
                        help='Number of segments. Default: {}'
                        .format(NUM_SEGMENTS))
    parser.add_argument('--compactness', type=float, default=COMPACTNESS,
                        help='Balances color proximity and space proximity. '
                        'A higher value gives more weight to space proximity '
                        'making superpixel shapes more square/cubic. Default: '
                        '{}'.format(COMPACTNESS))
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS,
                        help='Maximum number of iterations of k-means. '
                        'Default: {}'.format(MAX_ITERATIONS))
    parser.add_argument('--sigma', type=float, default=SIGMA,
                        help='Width of gaussian smoothing kernel for pre-'
                        'processing. Default: {}'.format(SIGMA))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--contour', dest='show_contour',
                       action='store_true', help='Shows a contour of the '
                       'superpixels in the output image. Default: {}'
                       .format(SHOW_CONTOUR))
    group.add_argument('--no-contour', dest='show_contour',
                       action='store_false', help='Doesn\'t show a contour of '
                       'the superpixels in the output image. Default: {}'
                       .format(not SHOW_CONTOUR))
    parser.set_defaults(show_contour=SHOW_CONTOUR)
    parser.add_argument('--contour-thickness', type=int,
                        default=CONTOUR_THICKNESS, help='The thickness of the '
                        'drawn contour. Default: {}'.format(CONTOUR_THICKNESS))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--center', dest='show_center',
                       action='store_true', help='Shows the center of mass '
                       'for all superpixels in the output image. Default: {}'
                       .format(SHOW_CENTER))
    group.add_argument('--no-center', dest='show_center',
                       action='store_false', help='Doesn\'t show the center '
                       'of mass for all superpixels in the output image. '
                       'Default: {}'.format(not SHOW_CENTER))
    parser.set_defaults(show_center=SHOW_CENTER)
    parser.add_argument('--center-radius', type=int, default=CENTER_RADIUS,
                        help='The radius of the drawn center. Default: {}'
                        .format(CENTER_RADIUS))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--mean', dest='show_mean', action='store_true',
                       help='Fills the superpixel with its mean color. '
                       'Default: {}'.format(SHOW_MEAN))
    group.add_argument('--no-mean', dest='show_mean',
                       action='store_false', help='Does\'t fill the '
                       'superpixel with its mean color. Default: {}'
                       .format(not SHOW_MEAN))
    parser.set_defaults(show_mean=SHOW_MEAN)

    args = parser.parse_args()

    # Give output fallback value dependent on input.
    if not args.output:
        path = args.image.split('/')
        name = path[-1].split('.')
        name = '{}-SLIC_zero.{}'.format('.'.join(name[:-1]), name[-1])
        args.output = '{}/{}'.format('/'.join(path[:-1]), name)

    return args


def main():
    """Runs the SLIC-zero superpixel segmentation."""

    args = get_arguments()

    # Calcuate the output directory and output filename from the ouput passed
    # as an argument.
    output = args.output.split('/')
    output_dir = '/'.join(output[0:-1]) if len(output) > 1 else '.'
    output_name = output[-1]

    # Load the image from file.
    image = cv2.imread(args.image)

    # Apply SLIC-zero.
    rep = image_to_slic_zero(image, args.num_segments,
                             compactness=args.compactness,
                             max_iterations=args.max_iterations,
                             sigma=args.sigma)

    # Extract the superpixels from the superpixel representation.
    superpixels = extract_superpixels(image, rep)

    # Save the superpixels to the output image.
    save_superpixel_image(image, superpixels, output_dir, output_name,
                          show_contour=args.show_contour,
                          contour_thickness=args.contour_thickness,
                          show_center=args.show_center,
                          center_radius=args.center_radius,
                          show_mean=args.show_mean)


# Only run if the script is executed directly.
if __name__ == '__main__':
    main()
