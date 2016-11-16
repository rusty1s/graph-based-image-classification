from __future__ import print_function

import argparse
import json

import tensorflow as tf

NETWORK_PARAMS = './network_params.json'
LEARNING_RATE = 1e-3


def get_arguments():
    parser = argparse.ArgumentParser(description='Training script for the '
                                     'Graph-based Image Classification')

    parser.add_argument('--network_params', type=str, default=NETWORK_PARAMS,
                        help='JSON file with the network parameters')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning raate for training')

    return parser.parse_args()


def main():
    args = get_arguments()

    # load the json network parameter from disk
    with open(args.network_params, 'r') as f:
        network_params = json.load(f)

# only execute if the script is executed directly
if __name__ == '__main__':
    main()
