from model import train

from data import Cifar10DataSet
cifar10 = Cifar10DataSet()

if __name__ == '__main__':
    train(cifar10, '/tmp/cifar10_train', 'network_params.json')
