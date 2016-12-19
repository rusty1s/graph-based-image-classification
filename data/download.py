import os
import sys
import urllib
import tarfile


def maybe_download_and_extract(url, data_dir, dirname, show_progress=True):
    """Downloads and extracts the given `url` to `data_dir`. Renames the
    extracted file to `dirname`."""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    dirpath = os.path.join(data_dir, dirname)

    if not os.path.exists(dirpath):
        def _progress(count, block_size, total_size):
            if not show_progress:
                return

            percent = count * block_size / total_size * 100.0

            sys.stdout.write(
                '\r>> Downloading {0:s} {1:.1f}%'.format(filename, percent)
            )
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        size = os.stat(filepath).st_size

        print()
        print('Successfully downloaded {} ({} bytes).'.format(filename, size))
        sys.stdout.write('Extracting {} to {}...'.format(filename, dirpath))
        sys.stdout.flush()

        with tarfile.open(filepath, 'r:gz') as f:
            extracted_dir = f.getnames()[0]
            f.extractall(data_dir)

        os.rename(os.path.join(data_dir, extracted_dir), dirpath)
        os.remove(filepath)

        print('Done!')
