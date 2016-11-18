"""SLIC Superpixel segmentation script.
"""

from __future__ import print_function

import argparse

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
# import matplotlib.pyplot as plt

# possible arguments (look up each help parameter for additional information)
SEGMENTS = 100
COMPACTNESS = 10
SIGMA = 1


def get_arguments():
    """Parses the arguments from the terminal and overrides their corresponding
    default values. Returns all arguments in a dictionary."""

    parser = argparse.ArgumentParser(description='Superpixel '
                                     'segmentation script')

    parser.add_argument('-i', '--image', required=True, type=str,
                        help='Path to the image')
    parser.add_argument('--segments', type=str, default=SEGMENTS,
                        help='Number of segments')
    parser.add_argument('--compactness', type=str, default=COMPACTNESS,
                        help='Balances color proximity and space proximity. '
                        'A higher value gives more weight to space proximity '
                        'making superpixel shapes more square/cubic.')
    parser.add_argument('--sigma', type=str, default=SIGMA,
                        help='Width of gaussian smoothing kernel for pre-'
                        'processing')
    parser.add_argument('-o', '--output', type=str,
                        help='Path and name to the superpixel segmented image')

    return parser.parse_args()


def main():
    """Runs the SLIC superpixel segmentation."""

    args = get_arguments()

    # load the image and convert it to a floating point data type
    # image = img_as_float(io.imread(args.image))

    # give output default value if none was passed
    if not args.output:
        path = args.image.split('/')
        name = path[-1].split('.')
        name = '{}-super.{}'.format('.'.join(name[:-1]), name[-1])
        output = '{}/{}'.format('/'.join(path[:-1]), name)
    else:
        output = args.output

    # apply SLIC and extract the supplied number of segments
    # segments = slic(image, n_segments=args.segments, sigma=5)

    # mark_boundaries(image, segments)
    # io.imsave('seg)-{}'.format(args.image), image)
    # print(segments)

#     # show the output of SLIC
#     fig = plt.figure('Superpixels -- {} segments'.format(args.segments))
#     ax = fig.add_subplit(1, 1, 1)
#     ax.imshow()
#     plt.axis('off')
    if not args.output:
        print('Output save in {}'.format(output))


# only run if the script is executed directly
if __name__ == '__main__':
    main()
