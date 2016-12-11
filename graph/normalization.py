import networkx as nx


def normalize(graph, neighborhood, node, labeling, size):
    """Normalizes the neighborhood of the graph with root node `node` to an
    unique order with size `size`."""

    subgraph = graph.subgraph(neighborhood)

    shortest_paths = nx.shortest_path(subgraph, source=node, weight=None)
    shortest_paths = {k: len(p) - 1 for k, p in shortest_paths.items()}

    # Swap the values and keys.
    result = {}
    for key, length in shortest_paths.items():
        if length in result:
            result[length].append(key)
        else:
            result[length] = [key]

    # TODO: Canonicaltion

    return []
