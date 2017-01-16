import tensorflow as tf
from skimage.future import graph


def spatial_adjacency(segmentation):
    def _spatial_adjacency(segmentation):
        pass

    return tf.py_func(_spatial_adjacency, [segmentation], tf.float32,
                      stateful=False, name='spatial_adjacency')
