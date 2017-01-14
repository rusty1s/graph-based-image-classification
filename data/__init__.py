from .dataset import DataSet

from .cifar_10 import Cifar10
from .pascal_voc import PascalVOC

from .helper.inputs import inputs
from .helper.tfrecord import (read_tfrecord, write_to_tfrecord)

datasets = {
    'cifar-10': Cifar10,
    'pascal-voc': PascalVOC,
}
