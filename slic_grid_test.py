from data import Cifar10
from data import PascalVOC
from segmentation import SlicGrid
from segmentation.algorithm import slic_generator
from segmentation.algorithm import slico_generator


SlicGrid(PascalVOC(), slic_generator(num_segments=400))
