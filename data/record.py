class Record(object):
    """A record object."""

    def __init__(self, data, shape, label):
        """Creates a record representing an example of a dataset.

        Args:
            data: A [height, width, depth] float32 tensor with the data of the
              record.
            shape: A TensorShape representing the shape of the data.
            label: An int64 tensor with the label of the record.
        """

        self._data = data
        self._data.set_shape(shape)

        self._shape = shape

        self._label = label
        self._label.set_shape([1])

    @property
    def data(self):
        """A [height, width, depth] float32 tensor with the data of the record.
        """
        return self._data

    @property
    def shape(self):
        """The shape of the data."""
        return self._shape

    @property
    def label(self):
        """An int64 tensor with the label of the record."""
        return self._label
