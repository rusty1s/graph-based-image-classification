from networkx import Graph


class SuperpixelGraph(Graph):
    def __init__(self, superpixels, node_mapping, edge_mapping):
        super(SuperpixelGraph, self).__init__()

        for s in superpixels:
            self.add_node(s.id, node_mapping(s))

            for id in s.neighbors:
                neighbor = [x for x in superpixels if x.id == id]
                self. add_edge(s.id, id, edge_mapping(s, neighbor[0]))
