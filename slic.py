"""SLIC Superpixel segmentation script
"""

from __future__ import print_function

import argparse

from src.superpixel import image_to_slic
from skimage.segmentation import mark_boundaries
# import matplotlib.pyplot as plt

# possible arguments (look up each help parameter for additional information)
SEGMENTS = 100
COMPACTNESS = 10.0
MAX_ITERATIONS = 10
SIGMA = 0.0


def get_arguments():
    """Parses the arguments from the terminal and overrides their corresponding
    default values. Returns all arguments in a dictionary."""

    parser = argparse.ArgumentParser(description='SLIC Superpixel '
                                     'segmentation script')

    parser.add_argument('-i', '--image', required=True, type=str,
                        help='Path to the image')
    parser.add_argument('--segments', type=str, default=SEGMENTS,
                        help='Number of segments')
    parser.add_argument('--compactness', type=float, default=COMPACTNESS,
                        help='Balances color proximity and space proximity. '
                        'A higher value gives more weight to space proximity '
                        'making superpixel shapes more square/cubic.')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS,
                        help='Maaximum number of iterations of k-means')
    parser.add_argument('--sigma', type=float, default=SIGMA,
                        help='Width of gaussian smoothing kernel for pre-'
                        'processing')
    parser.add_argument('-o', '--output', type=str,
                        help='Path and name to the superpixel segmented image')

    return parser.parse_args()


def main():
    """Runs the SLIC superpixel segmentation."""

    args = get_arguments()

    # give output default value if none was passed
    if not args.output:
        path = args.image.split('/')
        name = path[-1].split('.')
        name = '{}-super.{}'.format('.'.join(name[:-1]), name[-1])
        output = '{}/{}'.format('/'.join(path[:-1]), name)
    else:
        output = args.output

    # apply SLIC and extract the supplied number of segments
    segments = image_to_slic(args.image, segments=args.segments,
                             compactness=args.compactness,
                             max_iterations=args.max_iterations,
                             sigma=args.sigma)

    # mark_boundaries(image, segments)
    # io.imsave('seg)-{}'.format(args.image), image)
    # print(segments)

#     # show the output of SLIC
#     fig = plt.figure('Superpixels -- {} segments'.format(args.segments))
#     ax = fig.add_subplit(1, 1, 1)
#     ax.imshow()
#     plt.axis('off')
    if not args.output:
        print('Output saved in {}'.format(output))


# only run if the script is executed directly
if __name__ == '__main__':
    main()
