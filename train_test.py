import tensorflow as tf

from model import train
from data import Cifar10DataSet
from data import inputs


def main():
    cifar10 = Cifar10DataSet(data_dir='/tmp/cifar10_data')

    train(
        cifar10,
        train_dir='/tmp/cifar10_train',
        network_params_path='network_params_cifar10.json')


if __name__ == '__main__':
    main()
