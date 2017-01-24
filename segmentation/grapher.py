from .feature_extraction import feature_extraction
from segmentation import adjacencies


def grapher(algoritm, adjacency_name):
    def _to_graph(image):
        segmentation = algoritm_generator(image)
        nodes = feature_extraction(segmentation, image)
        adjacency = adjacencies[adjacency_name](segmentation)

        return nodes, adjacency

    return _to_graph
