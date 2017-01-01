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
            if not show_progress:
                return

            percent = count * block_size / total_size * 100.0

            sys.stdout.write(
                '\r>> Downloading {} {:.1f}%'.format(filename, percent)
            )
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        size = os.stat(filepath).st_size

        print()
        print('Successfully downloaded {} ({} bytes)'.format(filename, size))

    sys.stdout.write('>> Extracting {} to {}...'.format(filename, data_dir))
    sys.stdout.flush()

    tarfile.open(filepath, 'r:gz').extractall(data_dir)

    print('Done!')
