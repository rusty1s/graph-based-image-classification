import os

from cifar10 import Cifar10

cifar = Cifar10(os.path.join('.', 'datasets', 'cifar10'))
cifar.save_images()
