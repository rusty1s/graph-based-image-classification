import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Grapher(object):
    """Abstract class for defining a graph generator interface."""

    @property
    @abc.abstractmethod
    def num_node_channels(self):
        """The number of corresponding channels for each node in the graph.

        Returns:
            A number.
        """

        pass

    @property
    @abc.abstractmethod
    def num_edge_channels(self):
        """The number of corresponding channels for each edge in the graph.

        Returns:
            A number.
        """

        pass

    @abc.abstractmethod
    def create_graph(self, data):
        """Generates a graph based on the passed data.

        Args:
            data: A numpy array that holds the data.

        Returns:
            nodes: A numpy array that holds the channels for each node in the
              shape [num_nodes, num_node_channels].
            adjacencies: An numpy array that holds the (multiple) adjacency
              matrices of the graph in the shape
              [num_nodes, num_nodes, num_edge_channels].
        """

        pass
