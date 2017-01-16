import tensorflow as tf

from data import Cifar10
from data import PascalVOC
from data import iterator


def main(argv=None):
    pascal = PascalVOC()

    def each(output_batch, index, last_index):
        print(output_batch[1], index, last_index)

    iterate = iterator(pascal, eval_data=False, batch_size=10, num_epochs=1)
    iterate(each)


if __name__ == '__main__':
    tf.app.run()
