import os
import sys
import tarfile

from six.moves import urllib


def maybe_download_and_extract(url, data_dir, show_progress=True):
    """Downloads and extracts a tar file.

    Args:
        url: The url to download from.
        data_dir: The path to download to.

    Returns:
        The path to the extracted directory.
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    # Only download if file doesn't exist.
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        size = os.stat(filepath).st_size

        if show_progress:
            print('')

        print('Successfully downloaded {} ({} bytes).'.format(filename, size))

    mode = 'r:gz' if filename.split('.')[-1] == 'gz' else 'r'
    archive = tarfile.open(filepath, mode)

    # Get the top level directory in the tar file.
    extracted_dir = os.path.join(
        data_dir, os.path.commonprefix(archive.getnames()))

    # Only extract if file doesn't exist.
    if not os.path.exists(extracted_dir):
        sys.stdout.write(
            '>> Extracting {} to {}...'.format(filename, extracted_dir))
        sys.stdout.flush()

        tarfile.open(filepath, mode).extractall(data_dir)

        print('Done!')

    return extracted_dir


def _progress(count, block_size, total_size):
    """Prints a progress bar for downloads.

    Args:
        count: The number of blocks already downloaded.
        block_size: The size of a block.
        total_size: The total size of the file.
    """

    if show_progress:
        percent = 100.0 * count * block_size / total_size

        sys.stdout.write(
                '\r>> Downloading {} {:.1f}%'.format(filename, percent))
        sys.stdout.flush()
