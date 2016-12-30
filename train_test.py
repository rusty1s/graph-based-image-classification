from model import train
from data import (Cifar10DataSet, ConvertedDataSet)
from converter import PatchySan
from superpixel.algorithm import slico_generator
from grapher import SuperpixelGrapher
from data import inputs

import tensorflow as tf


def main():
    slico = slico_generator(
        num_superpixels=100,
        compactness=1.0,
        max_iterations=10,
        sigma=0.0)

    superpixel_grapher = SuperpixelGrapher(superpixel_algorithm=slico)

    patchy_san = PatchySan(
        grapher=superpixel_grapher,
        num_nodes=100,
        node_labeling='betweenness_centrality',
        node_stride=1,
        neighborhood_size=10,
        neighborhood_labeling='betweenness_centrality')

    cifar10 = Cifar10DataSet(data_dir='/tmp/cifar10_data')

    dataset = ConvertedDataSet(dataset=cifar10, converter=patchy_san)

    train(
        dataset,
        train_dir='/tmp/cifar10_train',
        network_params_path='network_params_slic.json')


if __name__ == '__main__':
    main()
