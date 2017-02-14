import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # nopep8
from sklearn.preprocessing import StandardScaler  # nopep8
from sklearn.decomposition import PCA  # nopep8
import tensorflow as tf  # nopep8
import numpy as np  # nopep8

from data import PascalVOC, read_tfrecord  # nopep8
from grapher import SegmentationGrapher  # nopep8
from segmentation.algorithm import slic_generator  # nopep8
from segmentation import adjacency_unweighted  # nopep8
from patchy import PatchySan  # nopep8


pascal = PascalVOC()
grapher = SegmentationGrapher(slic_generator(num_segments=400),
                              [adjacency_unweighted])
num_channels = grapher.num_node_channels
patchy = PatchySan(pascal, grapher,
                   data_dir='/tmp/patchy_san_slic_pascal_voc_data',
                   num_nodes=400, node_stride=1, neighborhood_size=1)

filename_queue = tf.train.string_input_producer(patchy.train_filenames,
                                                num_epochs=1, shuffle=False)

# Load the node features. We are not interested in the labels.
data, _ = read_tfrecord(filename_queue,
                        {'nodes': [-1, num_channels],
                         'neighborhood': [400, 1]})
data = data['nodes']

# The data queue.
data_batch = tf.train.batch(
    [data],
    batch_size=128,
    num_threads=16,
    capacity=300,
    dynamic_pad=True,
    allow_smaller_final_batch=True)
data_batch = tf.reshape(data_batch, [-1, num_channels])


sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

all_nodes = np.zeros((0, num_channels))

print('Analyzing dataset. This can take a few minutes.')

try:
    while(True):
        nodes = sess.run(data_batch)
        nodes = nodes[~np.all(nodes == 0, axis=1)]
        all_nodes = np.concatenate((all_nodes, nodes))

except KeyboardInterrupt:
    print('')

except tf.errors.OutOfRangeError:
    pass

finally:
    coord.request_stop()
    coord.join(threads)

    sess.close()

    pca = PCA()
    pca.fit(StandardScaler().fit_transform(all_nodes))

    # Build correlation coefficients for all node channels.
    cov = pca.get_covariance()
    plt.pcolor(cov)
    plt.colorbar()
    plt.yticks(np.arange(0.5, num_channels + 0.5, 5),
               np.arange(0, num_channels, 5))
    plt.xticks(np.arange(0.5, num_channels + 0.5, 5),
               np.arange(0, num_channels, 5))
    plt.savefig('covariance.svg', format='svg')
    plt.close()

    # Plot the cumulative explained variance ratio.
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig('pca.svg', format='svg')
    plt.close()
