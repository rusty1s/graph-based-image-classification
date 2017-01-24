import tensorflow as tf

from data import PascalVOC
from patchy import PatchySan
from grapher import SegmentationGrapher
from segmentation.algorithm import slic_generator
from segmentation import adjacency_euclidean_distance
from data import iterator


def main(argv=None):
    pascal = PascalVOC()
    grapher = SegmentationGrapher(
        slic_generator(300), adjacency_euclidean_distance)

    patchy = PatchySan(pascal, grapher, distort_inputs=True, num_nodes=300,
                       write_num_epochs=10)

if __name__ == '__main__':
    tf.app.run()
