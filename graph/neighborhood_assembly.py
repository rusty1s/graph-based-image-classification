def neighborhood_assembly(graph, node, size):
    """Assembles and returns a local neighborhood for the input node until
    there were at least `size` nodes found or until there are no more neighbors
    to add."""

    neighborhood = set([node])

    # We need to keep track of nodes, that freshly were added to the
    # neighborhood thus their neighborhood was not assembled yet.
    unnoted = set([node])

    # 1. There were at least `size` nodes found.
    # 2. There are no more neighbors to add.
    while len(neighborhood) < size and len(unnoted) > 0:

        # Keep track of the new neighborhoods and the new unnoted nodes.
        new_neighborhood = neighborhood.copy()
        new_unnoted = set()

        for v in unnoted:
            # Find neighborhood of visited node `v`.
            neighbors = set(graph.neighbors(v))

            # Add the nodes of the neighbors to the unnoted nodes set, which
            # aren't already in the tracked neighborhood.
            new_unnoted = new_unnoted.union(neighbors.difference(neighborhood))

            # Add found neighbors to the neighborhood set.
            new_neighborhood = new_neighborhood.union(neighbors)

        # Set the changed sets.
        neighborhood = new_neighborhood
        unnoted = new_unnoted

    return neighborhood
