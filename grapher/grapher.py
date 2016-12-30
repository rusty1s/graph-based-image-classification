import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Grapher(object):

    @property
    @abc.abstractmethod
    def node_channels_length(self):
        pass

    @abc.abstractmethod
    def create_graph(self, data):
        """Returns nodes and adjacent tensors."""
        pass
