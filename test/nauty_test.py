# from nose.tools import *
# from pynauty import Graph


# def test_graph():
#     # build the graph
#     g = Graph(10)

#     g.connect_vertex(0, [1, 2, 3, 4, 5])
#     g.connect_vertex(1, [0, 4, 5])
#     g.connect_vertex(2, [0, 3])
#     g.connect_vertex(3, [0, 2, 4, 6, 7])
#     g.connect_vertex(4, [0, 1, 3, 5, 6, 8, 9])
#     g.connect_vertex(5, [0, 1, 4])
#     g.connect_vertex(6, [3, 4, 7, 8])
#     g.connect_vertex(7, [3, 6, 8])
#     g.connect_vertex(8, [4, 6, 7, 9])
#     g.connect_vertex(9, [4, 8])

#   g.set_vertex_coloring([set([0]), set([1, 2, 3, 4, 5]), set([6, 7, 8, 9])])

#     # test the graph
#     assert_equal(g.number_of_vertices, 10)
#     assert_equal(g.directed, False)

#     assert_equal(g.adjacency_dict[0], [1, 2, 3, 4, 5])
#     assert_equal(g.adjacency_dict[1], [0, 4, 5])
#     assert_equal(g.adjacency_dict[2], [0, 3])
#     assert_equal(g.adjacency_dict[3], [0, 2, 4, 6, 7])
#     assert_equal(g.adjacency_dict[4], [0, 1, 3, 5, 6, 8, 9])
#     assert_equal(g.adjacency_dict[5], [0, 1, 4])
#     assert_equal(g.adjacency_dict[6], [3, 4, 7, 8])
#     assert_equal(g.adjacency_dict[7], [3, 6, 8])
#     assert_equal(g.adjacency_dict[8], [4, 6, 7, 9])
#     assert_equal(g.adjacency_dict[9], [4, 8])

#     assert_equal(g.vertex_coloring[0], set([0]))
#     assert_equal(g.vertex_coloring[1], set([1, 2, 3, 4, 5]))
#     assert_equal(g.vertex_coloring[2], set([6, 7, 8, 9]))
