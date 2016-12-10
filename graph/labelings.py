import networkx as nx


# Betweeness centrality of a node `v` is the sum of the fraction of all-pairs
# shortest paths that pass through `v`.
def betweenness_centrality(graph):
    """Computes the shortest-path betweeness centrality for nodes. Returns a
    list of all nodes in `graph` in descendant order corresponding to their
    betweeness centrality values."""

    # Returns a dictionary of nodes with betweeness centrality as the value
    result = nx.betweenness_centrality(graph, normalized=False)
    result = list(result.items())
    result = sorted(result, key=lambda v: v[1], reverse=True)

    return [v[0] for v in result]
