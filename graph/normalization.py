import networkx as nx


def normalize(graph, neighborhood, node, labeling, size):
    """Normalizes the neighborhood of the graph with root node `node` to an
    unique order with size `size`."""

    # We only look at the subgraph build py the passed neighborhood.
    subgraph = graph.subgraph(neighborhood)

    # Compute the shortest paths from every node in neighborhood to `node`.
    shortest_paths = nx.shortest_path(subgraph, source=node, weight=None)

    # We are only interested in the length of the paths.
    shortest_paths = {k: len(p) - 1 for k, p in shortest_paths.items()}

    # Swap the values and keys. We have the nodes now ordered by path length. A
    # path length can have multiple nodes.
    result = {}
    for key, length in shortest_paths.items():
        if length in result:
            result[length].append(key)
        else:
            result[length] = [key]

    # Order the nodes for a single path length according to the labeling.
    order = labeling(subgraph)

    for length, nodes in result.items():
        result[length] = [v for v in order if v in nodes]

    # TODO: Canonicaltion

    # Flatten the dictionary to one array.
    result = list(result.items())
    result = sorted(result, key=lambda i: i[0])
    result = [i[1] for i in result]
    result = [v for nodes in result for v in nodes]

    if len(result) >= size:
        return result[0:size]
    else:
        result.extend([None for _ in range(0, size - len(result))])
        return result
