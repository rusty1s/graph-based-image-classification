from __future__ import print_function

import os
import requests
from clint.textui import (progress, colored)
import tarfile
import numpy as np

# Load the correct pickle implementation for different python versions.
try:
    import cPickle as pickle
except:
    import _pickle as pickle


def extract_tar(tar, dir):
    """Extracts a tar file and moves it to `dir`."""

    # Extract the tar.gz.
    with tarfile.open(tar, 'r:gz') as f:
        extracted_dir = f.getnames()[0]
        f.extractall()

    # Move the extracted dir to the specified location.
    os.makedirs(dir)
    os.rename(extracted_dir, dir)

    # Remove tar file.
    os.remove(tar)


def download(url, file):
    """Downloads the given contens of the given `url` to `file` and prints a
    progress bar while waiting."""

    r = requests.get(url, stream=True)

    # Print a pretty progress bar while downloading.
    with open(file, 'wb') as f:
        total_length = r.headers.get('content-length')
        for chunk in progress.bar(r.iter_content(chunk_size=1024),
                                  expected_size=(int(total_length)/1024)):
            if chunk:
                f.write(chunk)
                f.flush()


def path_exists(path, action):
    """Checks if the given path exists and prints an error message if it
    doesn't based on the `action` one would want to perform."""

    if os.path.exists(path):
        print(colored.red('Abort {}:'.format(action)) +
              ' {} already exists.'.format(path))
        return True
    else:
        return False


def convert_1d_images_to_3d(batch, width, height, depth):
    reshaped = np.reshape(batch, (len(batch), depth, height, width))
    return np.transpose(reshaped, (0, 2, 3, 1))


def convert_batch(file, num_labels):
    """Converts a CIFAR-10 batch to a dictionary containing labels with the
    shape (10000, 10) and 3d images as data with the shape (10000, 32, 32, 3).
    """

    with open(file, 'rb') as f:
        # Decode to latin1 to avoid `UnicodeDecodeError`.
        batch = pickle.load(f, encoding='latin1')

    os.remove(file)

    with open(file, 'wb') as f:
        pickle.dump({
            'data': convert_image_batch(batch['data'], 32, 32, 3),
            'labels': batch['labels']
        }, f)
