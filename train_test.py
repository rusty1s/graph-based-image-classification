import tensorflow as tf

from data import Cifar10DataSet
from data import PatchySanDataSet
from model import train


def main():
    cifar10 = Cifar10DataSet(data_dir='/tmp/cifar10_data')
    patchy = PatchySanDataSet(dataset=cifar10)

    train(
        cifar10,
        train_dir='/tmp/cifar10_blala',
        network_params_path='network_params_slic.json')


if __name__ == '__main__':
    main()
