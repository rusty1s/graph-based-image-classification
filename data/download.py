import os
import sys
import tarfile

from six.moves import urllib


def maybe_download_and_extract(url, data_dir, show_progress=True):
    """Downloads and extracts the given url to data_dir."""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            if show_progress:
                percent = 100.0 * count * block_size / total_size

                sys.stdout.write(
                    '\r>> Downloading {} {:.1f}%'.format(filename, percent))
                sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        size = os.stat(filepath).st_size

        if show_progress:
            print()

        print('Successfully downloaded {} ({} bytes).'.format(filename, size))

    mode = 'r:gz' if filename.split('.')[-1] == 'gz' else 'r'
    archive = tarfile.open(filepath, mode)

    top_level_dir = os.path.commonprefix(archive.getnames())
    extracted_dir = os.path.join(data_dir, top_level_dir)

    if not os.path.exists(extracted_dir):
        sys.stdout.write(
            '>> Extracting {} to {}...'.format(filename, extracted_dir))
        sys.stdout.flush()

        tarfile.open(filepath, mode).extractall(data_dir)

        print('Done!')
