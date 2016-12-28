# import tensorflow as tf
# import networkx as nx

# from converter import (node_sequence, labeling)


# class PatchySanTest(tf.test.TestCase):

#     def test_node_sequence(self):
#         sequence = tf.constant([1, 2, 3, 4, 5, 6])

#         with self.test_session() as sess:
#             new_sequence = node_sequence(sequence, width=4, stride=1)
#             self.assertAllEqual(new_sequence.eval(), [1, 2, 3, 4])

#             new_sequence = node_sequence(sequence, width=3, stride=2)
#             self.assertAllEqual(new_sequence.eval(), [1, 3, 5])

#             new_sequence = node_sequence(sequence, width=4, stride=2)
#             self.assertAllEqual(new_sequence.eval(), [1, 3, 5, 0])

#             new_sequence = node_sequence(sequence, width=5, stride=4)
#             self.assertAllEqual(new_sequence.eval(), [1, 5, 0, 0, 0])

#     def test_labeling(self):
#         graph = nx.Graph()

#         graph.add_node(0)
#         graph.add_node(1)
#         graph.add_node(2)
#         graph.add_node(3)
#         graph.add_node(4)
#         graph.add_node(5)

#         graph.add_edge(0, 1)
#         graph.add_edge(1, 2)
#         graph.add_edge(2, 3)
#         graph.add_edge(2, 4)
#         graph.add_edge(3, 4)
#         graph.add_edge(3, 5)

#         with self.test_session() as sess:
#             sequence = labeling(graph)
#             self.assertAllEqual(sequence.eval(), [2, 1, 3, 0, 4, 5])


# if __name__ == '__main__':
#     tf.test.main()
