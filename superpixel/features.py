import tensorflow as tf
import numpy as np
from skimage.measure import regionprops


# NACHLESEN:
# Equivalent diameter WAS IST DAS?
# Euler number BRAUCHE ICH NICHT DA KEINE LOECHER???


def segmentation_features(segmentation):
    def _segmentation_features(segmentation):
        props = regionprops(segmentation)
        for i, prop in enumerate(props):
            print('----', i, '----')
            print('area', prop['area'])
            print('bbox', prop['bbox'])
            print('centroid', prop['local_centroid'])
            print('convex_area', prop['convex_area'])
            print('eccentricity', prop['eccentricity'])
            print('equivalent_diameter', prop['equivalent_diameter'])
            print('euler_number', prop['euler_number'])
            print('extent', prop['extent'])
            print('filled_area', prop['filled_area'])
            print('major_axis_length', prop['major_axis_length'])
            print('minor_axis_length', prop['minor_axis_length'])
            print('solidity', prop['solidity'])
            print('filled_area', prop['filled_area'])
            print('perimeter', prop['perimeter'])

            # print('moments', prop['moments']) liefert M00 bis M33
            # print('central moments', prop['moments_central'])

        return np.zeros((0, 0), dtype=np.float32)
        # return tf.zeros([2, 2], dtype=tf.float32)

    # We need to increment the segmentation, because labels with value 0 are
    # ignored when calling regionprops.
    segmentation = segmentation + tf.ones_like(segmentation)

    return tf.py_func(_segmentation_features, [segmentation], tf.float32,
                      stateful=False, name='segmentation_features')
