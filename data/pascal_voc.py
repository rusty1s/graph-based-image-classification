import os
import sys

from six.moves import xrange
from xml.dom.minidom import parse

import tensorflow as tf
from skimage.io import imread

from .dataset import DataSet
from .helper.download import maybe_download_and_extract
from .helper.tfrecord import read_tfrecord, write_to_tfrecord
from .helper.transform_image import crop_shape_from_box
from .helper.distort_image import distort_image_for_train,\
                                  distort_image_for_eval


DATA_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'\
           'VOCtrainval_11-May-2012.tar'
DATA_DIR = '/tmp/pascal_voc_data'

# The final shape of all images of the PascalVOC dataset.
HEIGHT = 224
WIDTH = 224

# Pass objects whose bounding boxes fall below a given bound.
MIN_OBJECT_HEIGHT = 50
MIN_OBJECT_WIDTH = 50

TRAIN_FILENAME = 'train.tfrecords'
TRAIN_INFO_FILENAME = 'train_info.txt'

EVAL_FILENAME = 'eval.tfrecords'
EVAL_INFO_FILENAME = 'eval_info.txt'


class PascalVOC(DataSet):
    """PascalVOC image classification dataset."""

    def __init__(self, data_dir=DATA_DIR, show_progress=True):
        """Creates a PascalVOC image classification dataset.

        Args:
            data_dir: The path to the directory where the PascalVOC dataset is
            stored.
            show_progress: Show a pretty progress bar for dataset computations.
        """

        super().__init__(data_dir, show_progress)

        # Download and extract dataset and write it to a tfrecord file.
        maybe_download_and_extract(DATA_URL, data_dir, show_progress)
        self._write_to_tfrecord()

        # PascalVOC does have a unique image size, but in the image
        # classification case we save multiple images for multiple objects for
        # one image. So we read the estimated number of examples per epoch from
        # an info file from disk.
        self._num_examples_per_epoch_for_train =\
            self._read_num_examples_per_epoch(
                os.path.join(data_dir, TRAIN_INFO_FILENAME))

        self._num_examples_per_epoch_for_eval =\
            self._read_num_examples_per_epoch(
                os.path.join(data_dir, EVAL_INFO_FILENAME))

    @property
    def train_filenames(self):
        """The filenames of the training batches from the PascalVOC dataset."""

        # All the training data is stored in a single TFRecord.
        return [os.path.join(self.data_dir, TRAIN_FILENAME)]

    @property
    def eval_filenames(self):
        """The filenames of the evaluation batches from the PascalVOC
        dataset."""

        # All the evaluation data is stored in a single TFRecord.
        return [os.path.join(self.data_dir, EVAL_FILENAME)]

    @property
    def labels(self):
        """The ordered labels of the PascalVOC dataset."""

        return ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
                'train', 'bottle', 'chair', 'diningtable', 'pottedplant',
                'sofa', 'tvmonitor']

    @property
    def num_examples_per_epoch_for_train(self):
        """The number of examples per epoch for training the PascalVOC dataset.
        """

        return self._num_examples_per_epoch_for_train

    @property
    def num_examples_per_epoch_for_eval(self):
        """The number of examples per epoch for evaluating the PascalVOC
        dataset."""

        return self._num_examples_per_epoch_for_eval

    def read(self, filename_queue):
        """Reads and parses examples from PascalVOC data files."""

        # Use the global reader operation for TFRecords.
        return read_tfrecord(filename_queue, [HEIGHT, WIDTH, 3])

    def distort_for_train(self, record):
        """Applies random distortions for training to a PascalVOC record."""

        return distort_image_for_train(record)

    def distort_for_eval(self, record):
        """Applies distortions for evaluation to a PascalVOC record."""

        return distort_image_for_eval(record)

    def _write_to_tfrecord(self):
        """Converts and writes the training and evaluation image sets to
        tfrecord files."""

        extracted_dir = os.path.join(self.data_dir, 'VOCdevkit', 'VOC2012')

        # Collect all relevant directories.
        image_sets_dir = os.path.join(extracted_dir, 'ImageSets', 'Main')
        image_dir = os.path.join(extracted_dir, 'JPEGImages')
        annotation_dir = os.path.join(extracted_dir, 'Annotations')

        # Write the training data.
        self._write_image_set_to_tfrecord(
            os.path.join(image_sets_dir, 'train.txt'), image_dir,
            annotation_dir, os.path.join(self.data_dir, TRAIN_FILENAME),
            os.path.join(self.data_dir, TRAIN_INFO_FILENAME))

        # Write the evaluation data.
        self._write_image_set_to_tfrecord(
            os.path.join(image_sets_dir, 'val.txt'), image_dir, annotation_dir,
            os.path.join(self.data_dir, EVAL_FILENAME),
            os.path.join(self.data_dir, EVAL_INFO_FILENAME))

    def _write_image_set_to_tfrecord(self, image_set_filename, image_dir,
                                     annotation_dir, tfrecord_filename,
                                     info_filename):
        """Converts and writes an image set to a tfrecord file.

        Args:
            image_set_filename: The filename containing the image names in the
              set, seperated in each line.
            image_dir: The directory containing the images.
            annotation_dir: The directory containing the annotations for all
              images.
            tfrecord_filename: The filename of the tfrecord file to save to.
            info_filename: The info filename to save the num examples per epoch
              information of the image set.
        """

        if tf.gfile.Exists(tfrecord_filename):
            return

        try:
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            # Read the lines of the image set filename which correspond to the
            # image names.

            with open(image_set_filename, 'r') as f:
                image_names = f.readlines()

            # Save statistics in variables.
            num_image_names = len(image_names)
            num_objects = 0
            num_objects_bypassed = 0

            # Iterate over all images and write them as a tfrecord to the
            # defined writer.
            for i in xrange(num_image_names):
                image_name = image_names[i].strip('\n')
                annotation_filename = os.path.join(annotation_dir,
                                                   '{}.xml'.format(image_name))
                image_filename = os.path.join(image_dir,
                                              '{}.jpg'.format(image_name))

                statistic = self._write_image_to_tfrecord(
                    writer, image_filename, annotation_filename)

                # Update the statistic.
                num_objects += statistic['num_objects']
                num_objects_bypassed += statistic['num_objects_bypassed']

                if self._show_progress:
                    percent = 100.0 * i / num_image_names

                    sys.stdout.write(
                        '\r>> Extracting objects to {} {:.1f}%'
                        .format(tfrecord_filename, percent))
                    sys.stdout.flush()

        except KeyboardInterrupt:
            pass

        finally:
            writer.close()

            self._write_num_examples_per_epoch(info_filename, num_objects)

            if self._show_progress:
                print('')

            print(' '.join([
                'Successfully extracted {} objects'.format(num_objects),
                'from {} images'.format(i),
                '({} bypassed).'.format(num_objects_bypassed),
            ]))

    def _write_image_to_tfrecord(self, writer, image_filename,
                                 annotation_filename):
        """Converts and expands an image to a tfrecord file.

        Args:
            writer: A TFRecordReader.
            image_filename: The filename of the image.
            annotation_filename: The filename to the annotaiton of the image.

        Returns:
            A stastic object containing the number of objects extracted from
            the image and the number of objects bypassed.
        """

        # Parse the xml annotation file and read the image into memory.
        image = imread(image_filename)
        annotation = parse(annotation_filename)

        num_objects = 0
        num_objects_bypassed = 0

        # Iterate over all objects in the image.
        for obj in annotation.getElementsByTagName('object'):
            # Bypass the objects that are either truncated or occluded.
            if int(_text_of_first_tag(obj, 'truncated')) > 0 or\
               int(_text_of_first_tag(obj, 'occluded')) > 0:
                num_objects_bypassed += 1
                continue

            # Extract the bounding box from the annotation object.
            bb_top = int(_text_of_first_tag(obj, 'ymin'))
            bb_right = int(_text_of_first_tag(obj, 'xmax'))
            bb_bottom = int(_text_of_first_tag(obj, 'ymax'))
            bb_left = int(_text_of_first_tag(obj, 'xmin'))

            # Check whether the bounding box is too small. If so, we discard
            # the object because it's irrelevant for classification tasks.
            if bb_bottom - bb_top < MIN_OBJECT_HEIGHT or\
               bb_right - bb_left < MIN_OBJECT_WIDTH:
                num_objects_bypassed += 1
                continue

            # The object on the image is valid for image classification.
            num_objects += 1

            # Finally crop it.
            cropped_image = crop_shape_from_box(
                image, [HEIGHT, WIDTH], [bb_top, bb_left, bb_bottom, bb_right])

            # Extract the label index from the annotation.
            label_name = _text_of_first_tag(obj, 'name')
            label_index = self.label_index(label_name)

            # Write the cropped image as a TFRecord example.
            write_to_tfrecord(writer, cropped_image, label_index)

        return {
            'num_objects': num_objects,
            'num_objects_bypassed': num_objects_bypassed,
        }

    def _write_num_examples_per_epoch(self, filename, num_examples_per_epoch):
        """Writes the number of examples per epoch to a filename.

        Args:
            filename: A tensor of type string.
            num_examples_per_epoch: An integer.
        """

        with open(filename, 'w') as f:
            f.write(str(num_examples_per_epoch))

    def _read_num_examples_per_epoch(self, filename):
        """Reads the number of examples per epoch of a filename.

        Args:
            filename: A tensor of type string.

        Returns:
            An integer.

        Raises:
            ValueError: If the filename doesn't exist.
        """

        if not tf.gfile.Exists(filename):
            raise ValueError('{} does not exist.'.format(filename))

        with open(filename, 'r') as f:
            return int(f.read())


def _text_of_first_tag(dom, tag):
    """Returns the text inside the first tag of the dom object.

    Args:
        dom: The dom object.
        tag: The tag name.

    Returns:
        A string.

    Raises:
        ValueError: If dom object doesn't contain the specified tag or if the
          first tag doesn't have a text.
    """

    tags = dom.getElementsByTagName(tag)

    # Tag not found.
    if len(tags) == 0 or tags[0].firstChild is None:
        raise ValueError('No tag {} found'.format(tag))

    # No text in first tag.
    if tags[0].firstChild is None:
        raise ValueError('No text in tag {} found'.format(tag))

    return dom.getElementsByTagName(tag)[0].firstChild.nodeValue
