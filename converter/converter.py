import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Converter(object):

    @property
    @abc.abstractmethod
    def shape(self):
        pass

    @abc.abstractmethod
    def convert(self, data):
        pass
