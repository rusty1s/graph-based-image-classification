import networkx as nx


def graph_from_segments(segments, node_mapping, edge_mapping):
    G = nx.Graph()

    for id in segments:
        s = segments[id]
        G.add_node(id, node_mapping(s))

        for neighbor_id in s.neighbors:
            G.add_edge(id, neighbor_id, edge_mapping(s, segments[neighbor_id]))

    return G
