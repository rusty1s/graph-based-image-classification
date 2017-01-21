from .dataset import DataSet
from .helper.record import Record

from .cifar_10 import Cifar10
from .pascal_voc import PascalVOC

from .helper.inputs import inputs
from .helper.iterator import iterator
from .helper.tfrecord import read_tfrecord, write_to_tfrecord
from .helper.distort_image import distort_image_for_train,\
                                  distort_image_for_eval

datasets = {
    'cifar_10': Cifar10,
    'pascal_voc': PascalVOC,
}
