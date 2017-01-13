import os
import sys

import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from xml.dom.minidom import parse

from .dataset import DataSet
from .download import maybe_download_and_extract
from .tfrecord import (tfrecord_example, read_tfrecord)


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
    """PascalVOC dataset."""

    def __init__(self, data_dir=DATA_DIR):

        super().__init__(data_dir)

        maybe_download_and_extract(DATA_URL, data_dir)
        self._save_as_tfrecord()

        with open(os.path.join(data_dir, TRAIN_INFO_FILENAME), 'r') as f:
            self._num_examples_per_epoch_for_train = int(f.readline())

        with open(os.path.join(data_dir, EVAL_INFO_FILENAME), 'r') as f:
            self._num_examples_per_epoch_for_eval = int(f.readline())

    @property
    def train_filenames(self):
        return [os.path.join(self.data_dir, TRAIN_FILENAME)]

    @property
    def eval_filenames(self):
        return [os.path.join(self.data_dir, EVAL_FILENAME)]

    @property
    def labels(self):
        return ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
                'train', 'bottle', 'chair', 'diningtable', 'pottedplant',
                'sofa', 'tvmonitor']

    @property
    def num_examples_per_epoch_for_train(self):
        return self._num_examples_per_epoch_for_train

    @property
    def num_examples_per_epoch_for_eval(self):
        return self._num_examples_per_epoch_for_eval

    def read(self, filename_queue):
        return read_tfrecord(filename_queue, [HEIGHT, WIDTH, 3])

    def _save_as_tfrecord(self):
        extracted_dir = os.path.join(self.data_dir, 'VOCdevkit', 'VOC2012')

        # Collect all relevant directories.
        image_sets_dir = os.path.join(extracted_dir, 'ImageSets', 'Main')
        image_dir = os.path.join(extracted_dir, 'JPEGImages')
        annotation_dir = os.path.join(extracted_dir, 'Annotations')

        # Save the training data.
        self._save_image_set_as_tfrecord(
            os.path.join(image_sets_dir, 'train.txt'), image_dir,
            annotation_dir, os.path.join(self.data_dir, TRAIN_FILENAME),
            os.path.join(self.data_dir, TRAIN_INFO_FILENAME))

        # Save the evaluation data.
        self._save_image_set_as_tfrecord(
            os.path.join(image_sets_dir, 'val.txt'), image_dir, annotation_dir,
            os.path.join(self.data_dir, EVAL_FILENAME),
            os.path.join(self.data_dir, EVAL_INFO_FILENAME))

    def _save_image_set_as_tfrecord(self, image_set_filename, image_dir,
                                    annotation_dir, tfrecord_filename,
                                    info_filename, show_progress=True):

        if os.path.exists(tfrecord_filename):
            return

        try:
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            # Read the lines of the image set filename which correspond to the
            # image names.
            f = open(image_set_filename)
            image_names = f.readlines()

            # Save statistics in variables.
            num_image_names = len(image_names)
            num_objects = 0
            num_objects_bypassed = 0

            # Iterate over all images and save them as a TFRecord to the
            # defined writer.
            for i, image_name in enumerate(image_names):
                image_name = image_name.strip('\n')
                statistic = self._save_image_by_writer(
                    writer, image_name, image_dir, annotation_dir, i,
                    num_image_names, tfrecord_filename, show_progress)

                # Update the statistic.
                num_objects += statistic['num_objects']
                num_objects_bypassed += statistic['num_objects_bypassed']

        except KeyboardInterrupt:
            pass

        finally:
            writer.close()
            f.close()

            with open(info_filename, 'w') as f:
                f.write(str(num_objects))

            if show_progress:
                print('')

            print(' '.join([
                'Successfully extracted {} objects'.format(num_objects),
                'from {} images'.format(i),
                '({} bypassed).'.format(num_objects_bypassed),
            ]))

    def _save_image_by_writer(self, writer, image_name, image_dir,
                              annotation_dir, current_index, count,
                              tfrecord_filename, show_progress):

        # Parse the xml annotation file and read the image into memory.
        annotation = parse(os.path.join(annotation_dir,
                                        '{}.xml'.format(image_name)))
        image = imread(os.path.join(image_dir, '{}.jpg'.format(image_name)))

        num_objects = 0
        num_objects_bypassed = 0

        # Iterate over all objects in the image.
        for obj in annotation.getElementsByTagName('object'):

            # Bypass the objects that are either truncated or occluded.
            if int(_text_of_first_tag(obj, 'truncated')) > 0 or\
               int(_text_of_first_tag(obj, 'occluded')) > 0:
                num_objects_bypassed += 1
                continue

            cropped_image = self._crop_and_rescale_image(image, obj)

            if cropped_image is None:
                # Bypass the image if the bounding box is too small.
                num_objects_bypassed += 1
                continue
            else:
                num_objects += 1

                # Extract the label index from the annotation.
                label_name = _text_of_first_tag(obj, 'name')
                label_index = self.label_index(label_name)

                # Save the cropped image as a TFRecord example.
                cropped_image = cropped_image.astype(np.float32)
                example = tfrecord_example(cropped_image, label_index)
                writer.write(example.SerializeToString())

        if show_progress:
            sys.stdout.write(
                '\r>> Extracting objects to {} {:.1f}%'
                .format(tfrecord_filename, 100.0 * current_index / count))
            sys.stdout.flush()

        return {
            'num_objects': num_objects,
            'num_objects_bypassed': num_objects_bypassed,
        }

    def _crop_and_rescale_image(self, image, obj):
        # Extract the bounding box from the annotation object.
        bb_top = int(_text_of_first_tag(obj, 'ymin'))
        bb_height = int(_text_of_first_tag(obj, 'ymax')) - bb_top
        bb_left = int(_text_of_first_tag(obj, 'xmin'))
        bb_width = int(_text_of_first_tag(obj, 'xmax')) - bb_left

        # Check whether the bounding box is too small. If so, we discard the
        # object because it's irrelevant for classification tasks.
        if bb_height < MIN_OBJECT_HEIGHT or bb_width < MIN_OBJECT_WIDTH:
            return None

        # Crop the bounding box or the maximal defined resolution of the image,
        # whichever resolution is greater, so that we always crop the full
        # bounding box of the object into the image.
        height = max(bb_height, HEIGHT)
        width = max(bb_width, WIDTH)

        # Crop the image based on the center of the bounding box.
        crop_top = max(bb_top + (bb_height - height) // 2, 0)
        crop_left = max(bb_left + (bb_width - width) // 2, 0)

        # We need to adjust the variables if the object is too far at the right
        # or the bottom of the image, so that we can get the maximal cropping
        # defined by the height and width.
        crop_top = min(crop_top, max(image.shape[0] - height, 0))
        crop_left = min(crop_left, max(image.shape[1] - width, 0))

        # Calculate the opposite sides of the cropping in case the image is
        # smaller than the defined cropping.
        crop_bottom = min(crop_top + height, image.shape[0])
        crop_right = min(crop_left + width, image.shape[1])

        # Finally crop the image.
        image = image[crop_top:crop_bottom, crop_left:crop_right]

        # Rescale the image if needed and return.
        return self._rescale(image)

    def _rescale(self, image):
        # The passed image can be greater or smaller than the wished fixed
        # resolution. We need to either scale the image up or down and crop it
        # again if this is the case.
        if image.shape[0] == HEIGHT and image.shape[1] == WIDTH:
            # Nothing to do here.
            return image

        if image.shape[0] < HEIGHT or image.shape[1] < WIDTH:
            # Scale up.
            scale = max(1.0 * HEIGHT / image.shape[0],
                        1.0 * WIDTH / image.shape[1])

        else:
            # Scale down.
            scale = min(1.0 * image.shape[0] / HEIGHT,
                        1.0 * image.shape[1] / WIDTH)

        # Calculate the shape after resizing and resize the image based on this
        # shape.
        shape = [max(int(scale * image.shape[0]), HEIGHT),
                 max(int(scale * image.shape[1]), WIDTH)]
        image = resize(image, shape, preserve_range=True)

        # Finally crop the image again based on its center.
        crop_top = (image.shape[0] - HEIGHT) // 2
        crop_bottom = crop_top + HEIGHT
        crop_left = (image.shape[1] - WIDTH) // 2
        crop_right = crop_left + WIDTH

        return image[crop_top:crop_bottom, crop_left:crop_right]


def _text_of_first_tag(dom, tag):
    return dom.getElementsByTagName(tag)[0].firstChild.nodeValue
