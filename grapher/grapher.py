import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Grapher(object):

    @property
    @abc.abstractmethod
    def num_node_channels(self):
        pass

    @property
    @abc.abstractmethod
    def num_adjacency_matrices(self):
        pass

    @abc.abstractmethod
    def create_graph(self, data):
        pass
