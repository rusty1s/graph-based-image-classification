#!/bin/sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/.miniconda"
rm -f miniconda.sh
export PATH="$HOME/.miniconda/bin:$PATH"
conda config --set always_yes yes

source bin/install.sh -n test --python $TRAVIS_PYTHON_VERSION --tensorflow $TENSORFLOW --nauty $NAUTY --pynauty $PYNAUTY
