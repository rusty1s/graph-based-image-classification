from .dataset import DataSet

from .cifar_10 import Cifar10
from .pascal_voc import PascalVOC

from .inputs import inputs
from .tfrecord import (tfrecord_example, read_tfrecord)

datasets = {
    'cifar-10': Cifar10,
    'pascal-voc': PascalVOC,
}
