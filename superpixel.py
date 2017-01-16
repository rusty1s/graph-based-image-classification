import tensorflow as tf

from data import Cifar10
from data import PascalVOC
from data import iterator

from superpixel.algorithm import slic_generator


def main(argv=None):
    pascal = PascalVOC()
    slic = slic_generator(10)

    def _before(image_batch, label_batch):
        image = tf.squeeze(image_batch, squeeze_dims=[0])

        return [image, slic(image)]

    def _each(output, index, last_index):
        if index == 1:
            print(output[0])
        else:
            print(index / last_index)

        # print(output_batch, index, last_index)

    iterate = iterator(pascal, eval_data=False, batch_size=1,
                       distort_inputs=True)
    iterate(_each, _before)


if __name__ == '__main__':
    tf.app.run()
