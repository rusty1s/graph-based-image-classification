from model import train

from data import (Cifar10DataSet, ConvertedDataSet)
from converter import PatchySan

cifar10 = Cifar10DataSet()
patchy_san = PatchySan(100, 'order', 1, 10, 'betwenness_centrality', 4)

dataset = ConvertedDataSet(cifar10, patchy_san)

if __name__ == '__main__':
    train(dataset, '/tmp/cifar10_train', 'network_params_slic.json')
