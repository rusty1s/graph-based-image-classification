import os

import tensorflow as tf

from .inputs import inputs


class InputTest(tf.test.TestCase):

    def _record(self, label, red, green, blue):
        image_size = 32 * 32
        return bytearray([label] + [red, green, blue] * image_size)

    def _write_records(name):

        # Create three records.
        labels = [9, 3, 0]
        records = [self._record(labels[0], 0, 128, 255),
                   self._reocrd(labels[1], 255, 0, 1),
                   self._record(labels[2], 254, 255, 0)]

        # Write the records to `/tmp/{name}`.
        contents = bytearray([record for record in records])
        filename = os.path.join(self.get_temp_dir(), name)
        open(filename, 'wb').write(contents)

#         with self.test_session() as sess:
#             q = tf.FIFOQueue(00, [tf.string], shapes=())

    def _read():
        pass


# inputs(data_dir, filenames, read, num_examples_per_epoch, batch_size,
# preprocess=None, shuffle=True):
