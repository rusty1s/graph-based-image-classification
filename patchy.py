import tensorflow as tf
import numpy as np

from data import iterator
from data import PascalVOC
from patchy import PatchySan
from grapher import SegmentationGrapher

pascal = PascalVOC()
grapher = SegmentationGrapher(None, None)
patchy = PatchySan(pascal, grapher, num_nodes=440,
                   data_dir='/tmp/patchy_san_slic_pascal_voc_data')

iterate = iterator(patchy, False, zero_mean_inputs=False, shuffle=True)


def _before(data, label):
    # image = tf.cast(data, dtype=tf.uint8)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.image.rgb_to_hsv(image)
    # image = tf.strided_slice(image, [0, 0, 2], [224, 224, 3], [1, 1, 1])
    # image = tf.squeeze(image)
    data = tf.strided_slice(data, [0, 0], [300, 40], [1, 1])
    return [data, label]


# 44, 46 und 45 bis 55 bis jetz muessen raus, frage ist warum
def _each(output, index, last_index):
    if (index > 1):
        return
    # 440 * 9 * 83
    data = output[0]
    cov = np.corrcoef(data, rowvar=False)
    np.savetxt('bla.txt', cov, fmt='%.1f', delimiter=', ')
    print(cov)


iterate(_each, _before)
