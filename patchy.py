import tensorflow as tf

from data import PascalVOC
from patchy import PatchySan
from segmentation import SegmentationGrapher
from segmentation.algorithm import slic_generator
from data import iterator


def main(argv=None):
    pascal = PascalVOC()
    grapher = SegmentationGrapher(slic_generator(289), 'euclid_distance')

    patchy = PatchySan(pascal, grapher, distort_inputs=True, num_nodes=300)


if __name__ == '__main__':
    tf.app.run()
