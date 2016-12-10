import networkx as nx


def betweenness_centrality(graph):
    """Computes the shortest-path betweenness centrality for nodes. Betweenness
    centrality of a node `v` is the sum of the fraction of all-pairs shortest
    paths that pass through `v`. Returns a list of all nodes in `graph` in
    descendant order corresponding to their betweenness centrality values."""

    # Returns a dictionary of nodes with betweeness centrality as the value
    result = nx.betweenness_centrality(graph, normalized=False)
    result = list(result.items())
    result = sorted(result, key=lambda v: v[1], reverse=True)

    return [v[0] for v in result]


def order(graph):
    """If the graph nodes have an actual attribute `order`, it is used to
    calculate the order of the graph."""

    print(graph.nodes)
    return []
