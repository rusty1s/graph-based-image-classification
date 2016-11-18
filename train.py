"""Training script for the Graph-based Image Classification
"""

from __future__ import print_function

import argparse
import json

import tensorflow as tf
from pynauty import Graph

from src.helper import io

# possible arguments (look up each help parameter for additional information)
DATA = '../data'
NETWORK_PARAMS = '../network_params.json'
LEARNING_RATE = 1e-3
EPOCHS = int(1e5)  # 100.000
CHECKPOINT_STEP = 50


def get_arguments():
    """Parses the arguments from the terminal and overrides their corresponding
    default values. Returns all arguments in a dictionary."""

    parser = argparse.ArgumentParser(description='Training script for the '
                                     'Graph-based Image Classification')

    parser.add_argument('--data', type=str, default=DATA,
                        help='The directory containing the training samples')
    parser.add_argument('--network_params', type=str, default=NETWORK_PARAMS,
                        help='JSON file with the network parameters')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training steps')
    parser.add_argument('--checkpoint_step', type=int,
                        default=CHECKPOINT_STEP,
                        help='How many steps to save each checkpoint after')

    return parser.parse_args()


def main():
    """Runs the training."""

    args = get_arguments()

    # load the json network parameter from disk
    with open(args.network_params, 'r') as f:
        network_params = json.load(f)


# only run if the script is executed directly
if __name__ == '__main__':
    main()
