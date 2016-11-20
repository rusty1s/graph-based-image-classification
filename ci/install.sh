#!/bin/sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/.miniconda"
export PATH="$HOME/.miniconda/bin:$PATH"
conda config --set always_yes yes

# create test environement
conda create -q -n test python=$TRAVIS_PYTHON_VERSION numpy
source activate test

# install codecov for code coverage
pip install codecov

# install TensorFlow
if [[ $TRAVIS_PYTHON_VERSION == "2.7" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp27-none-linux_x86_64.whl
elif [[ $TRAVIS_PYTHON_VERSION == "3.4" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp34-cp34m-linux_x86_64.whl
elif [[ $TRAVIS_PYTHON_VERSION == "3.5" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp35-cp35m-linux_x86_64.whl
fi

# install nauty and its python wrapper pynauty
curl https://web.cs.dal.ca/~peter/software/pynauty/pynauty-$PYNAUTY.tar.gz | tar xz
curl http://users.cecs.anu.edu.au/~bdm/nauty/nauty$NAUTY.tar.gz | tar xz

mv nauty$NAUTY pynauty-$PYNAUTY/nauty
mkdir -p .sources
mv pynauty-$PYNAUTY .sources/pynauty

make pynauty -C .sources/pynauty
make user-ins -C .sources/pynauty

# install requirements
pip install -r requirements.txt
pip install -r requirements_test.txt
