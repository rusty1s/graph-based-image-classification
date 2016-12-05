"""Download script for the CIFAR-10 train and test images.
"""

import os
import argparse

from cifar10 import Cifar10

# possible arguments (look up each help parameter for additional information)
OUTPUT = os.path.join('.', 'datasets', 'cifar-10')


def get_arguments():
    """Parses the arguments from the terminal and overrides their corresponding
    default values. Returns all arguments in a dictionary."""

    parser = argparse.ArgumentParser(description='Download script for the '
                                     'CIFAR-10 train and test images')

    parser.add_argument('-o', '--output', type=str, default=OUTPUT,
                        help='Output directory in which to save the CIFAR-10 '
                        'images. Must not exist! Default: ./datasets/cifar-10')

    return parser.parse_args()


def main():
    """Runs the downloader."""

    args = get_arguments()

    Cifar10(args.output).save_images()


# only run if the script is executed directly
if __name__ == '__main__':
    main()
