from model import train
from data import (Cifar10DataSet, ConvertedDataSet)
from converter import PatchySan
from superpixel.algorithm import slico_generator
from grapher import SuperpixelGrapher

if __name__ == '__main__':
    superpixel_grapher = SuperpixelGrapher(slico_generator)

    patchy_san = PatchySan(
        grapher=superpixel_grapher,
        num_nodes=100,
        node_labeling='betweenness_centrality',
        node_stride=1,
        neighborhood_size=10,
        neighborhood_labeling='betweenness_centrality')

    cifar10 = Cifar10DataSet()

    dataset = ConvertedDataSet(cifar10, patchy_san)
    # train(dataset, '/tmp/cifar10_train', 'network_params.json')
