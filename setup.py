from setuptools import setup
from setuptools import find_packages

setup(name='pgcn',
      version='1.0',
      description='Planar Graph Convolutional Networks in TensorFlow',
      author='Matthias Fey',
      author_email='matthias.fey@tu-dortmund.de',
      url='https://github.com/rusty1s/deep-learning/thesis',
      download_url='https://github.com/rusty1s/pgcn',
      license='MIT',
      install_requires=['numpy',
                        'scipy',
                        'networkx',
                        'tensorflow',
                        'scikit-learn',
                        'scikit-image',
                        ],
      package_data={'pgcn': ['README.md']},
      packages=find_packages())
