import tensorflow as tf
import networkx as nx
import numpy as np

from superpixel import (Superpixel, extract)

from .grapher import Grapher


def _node_channels(superpixel):
    color = superpixel.mean
    center = superpixel.relative_center_in_bounding_box

    return [
        color[0],
        color[1],
        color[2],
        superpixel.count,
        center[1],
        center[0],
        superpixel.height,
        superpixel.width,
    ]


class SuperpixelGrapher(Grapher):

    def __init__(self, superpixel_algorithm):
        self._superpixel_algorithm = superpixel_algorithm

    @property
    def node_channels_length(self):
        return len(_node_channels(Superpixel()))

    def create_graph(self, image):
        segmentation = self._superpixel_algorithm(image)
        num_segments = tf.reduce_max(segmentation) + 1
        values = tf.range(0, num_segments, dtype=tf.int32)

        ones = tf.ones_like(segmentation)
        zeros = tf.zeros_like(segmentation)

        def _bounding_box(mask):
            cols = tf.reduce_any(mask, axis=0)
            cols = tf.where(cols)
            cols = tf.reshape(cols, [-1])
            rows = tf.reduce_any(mask, axis=1)
            rows = tf.where(rows)
            rows = tf.reshape(rows, [-1])
            top = tf.strided_slice(rows, [0], [1], [1])
            bottom = tf.strided_slice(tf.reverse(rows, [True]), [0], [1], [1]) + 1
            left = tf.strided_slice(cols, [0], [1], [1])
            right = tf.strided_slice(tf.reverse(cols, [True]), [0], [1], [1]) + 1
            return top, right, bottom, left

        def _extract(value):
            mask = tf.equal(segmentation, tf.scalar_mul(value, ones))

            # get the bounding box
            top, right, bottom, left = _bounding_box(mask)

            # slice the mask
            mask = tf.strided_slice(mask, tf.concat(0, [top, left]),
                                    tf.concat(0, [bottom, right]), [1, 1])

            # calculate center
            indices = tf.where(mask)
            indices = tf.transpose(indices)
            indices = tf.cast(indices, tf.float32)
            center = tf.reduce_mean(indices, axis=1)

            # calculate count
            mask = tf.cast(mask, tf.int8)
            count = tf.count_nonzero(mask, dtype=tf.float32)

            features = tf.concat(0, [
                [count],
                tf.strided_slice(center, [0], [1], [1]),
                tf.strided_slice(center, [1], [2], [1]),
                tf.cast(right - left, tf.float32),
                tf.cast(bottom - top, tf.float32),
            ])
            return features

        nodes = tf.map_fn(
            _extract, values, dtype=tf.float32, parallel_iterations=100)
        return nodes

        return nodes, tf.ones([100, 100])

        


        # segmentation = tf.reshape(segmentation, [-1])
        # return segmentation
        # image = tf.reshape(image, [-1, 3])
        # segmentation = tf.reshape()

        # return tf.segment_sum(image, segmentation)
        
        # num_superpixels = tf.reduce_max(superpixel_representation) + 1
        # superpixels = tf.range(0, num_superpixels, dtype=tf.int32)

        # tf.map_fn()
        # # return num_superpixels
        # return superpixels


        

        # def _create_graph(image, superpixel_representation):

        #     superpixels = extract(image, superpixel_representation)

        #     graph = nx.Graph()

        #     def _node_mapping(superpixel):
        #         return {'features': _node_channels(superpixel)}

        #     for s in superpixels:
        #         graph.add_node(s.id, _node_mapping(s))

        #         for id in s.neighbors:
        #             graph.add_edge(s.id, id)

        #     nodes = list(nx.get_node_attributes(graph, 'features').items())
        #     nodes = sorted(nodes, key=lambda v: v[0])
        #     nodes = [v[1] for v in nodes]
        #     nodes = np.array(nodes, dtype=np.float32)

        #     adjacent = nx.to_numpy_matrix(graph)
        #     adjacent = adjacent.astype(np.float32)

        #     return nodes, adjacent

        # # Extract is bottleneck, makes the shit 10 times slower TODO
        # superpixel_representation = self._superpixel_algorithm(image)
        # # superpixel = tf.strided_slice(superpixel_representation, [0, 0], [100, 8], [1, 1])
        # # superpixel = tf.cast(superpixel, tf.float32)
        # # return superpixel, tf.ones([100, 100])
        # return tf.py_func(_create_graph, [image, superpixel_representation],
        #                   [tf.float32, tf.float32], stateful=False,
        #                   name='superpixel_grapher')
