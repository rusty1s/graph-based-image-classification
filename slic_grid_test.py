from data import Cifar10
from data import PascalVOC
from superpixel import SlicGrid
from superpixel.algorithm import slic_generator
from superpixel.algorithm import slico_generator


SlicGrid(PascalVOC(), slic_generator(num_superpixels=400))
