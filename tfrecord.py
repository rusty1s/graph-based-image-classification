from data import write
from data import Cifar10DataSet


if __name__ == '__main__':
    dataset = Cifar10DataSet(data_dir='/tmp/cifar10_data')
    filename = '/tmp/citrain.tfrecords'

    write(
        dataset=dataset,
        filename=filename,
        train=False,
        eval_data=True,
        epochs=2,
        batch_size=100)
