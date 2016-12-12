from networkx import Graph


def create_superpixel_graph(superpixels, node_mapping, edge_mapping):
    graph = Graph()

    for s in superpixels:
        graph.add_node(s.id, node_mapping(s))

        for id in s.neighbors:
            neighbor = [x for x in superpixels if x.id == id]
            graph.add_edge(s.id, id, edge_mapping(s, neighbor[0]))

    return graph
