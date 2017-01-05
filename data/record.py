class Record(object):

    def __init__(self, height, width, depth, label, data):
        """Creates a record defining an example of a dataset.

        Args:
            height: Number of rows in the record.
            width: Number of columns in the record.
            depth: Number of channels in the record.
            label: An int64 tensor with the label of the record.
            data: A [height, width, depth] float32 tensor with the data of the
              record.
        """

        self._height = height
        self._width = width
        self._depth = depth

        self._label = label
        self._label.set_shape([1])

        self._data = data
        self._data.set_shape([height, width, depth])

    @property
    def height(self):
        """The number of rows in the record."""
        return self._height

    @property
    def width(self):
        """The number of columns in the record."""
        return self._width

    @property
    def depth(self):
        """The number of channels in the record."""
        return self._depth

    @property
    def label(self):
        """A int64 tensor with the label of the record."""
        return self._label

    @property
    def data(self):
        """A [height, width, depth] float32 tensor with the data of the record.
        """
        return self._data
