import tensorflow as tf

from superpixel import create_graph


# class GraphTest(tf.test.TestCase):

#     def test_create_graph(self):
#         image = tf.constant([])

#         superpixels = tf.constant([
#             [1, 1, 2, 2],
#             [1, 1, 2, 2],
#             [3, 3, 4, 4],
#             [3, 3, 4, 4],
#         ])

#         with self.test_session() as sess:
#             nodes, adjacent = create_graph(image, superpixels)
#             print(nodes.eval())
#             print(adjacent.eval())
