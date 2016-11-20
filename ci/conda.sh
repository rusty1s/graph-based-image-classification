#!/bin/bash

while [[ $# -gt 1 ]]; do
  key="$1"

  case $key in
    -v|--version)
      VERSION="$2"
      shift
      ;;
    -n|--name)
      NAME="$2"
      shift
      ;;
    --tensorflow)
      TENSORFLOW="$TENSORFLOW"
      shift
      ;;
    --nauty)
      NAUTY="$NAUTY"
      shift
      ;;
    --pynauty)
      PYNAUTY="$PYNAUTY"
      shift
      ;;
    *)
      ;;
  esac

  shift
done

# set default values
VERSION="${VERSION:-3.5}"
TENSORFLOW="${TENSORFLOW:-0.11.0}"
NAUTY="${NAUTY:-26r7}"
PYNAUTY="${PYNAUTY:-0.6.0}"

if [[ -z "$NAME" ]]; then
  echo "Must provide a --name argument."
  exit 1
fi

if ! type conda > /dev/null; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/.miniconda"
  export PATH="$HOME/.miniconda/bin:$PATH"
fi

# create conda environement
conda create -q -n "$NAME" python=$VERSION
source activate "$NAME"

# install conda requirements
conda install numpy scipy matplotlib
conda install -c https://conda.binstar.org/menpo opencv3

# install codecov for code coverage
pip install codecov

# install TensorFlow
if [[ $VERSION == "2.7" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp27-none-linux_x86_64.whl
elif [[ $VERSION == "3.4" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp34-cp34m-linux_x86_64.whl
elif [[ $VERSION == "3.5" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp35-cp35m-linux_x86_64.whl
fi

# install nauty and its python wrapper pynauty
if [[ ! -d "$HOME/.sources/pynauty" ]]; then
  curl https://web.cs.dal.ca/~peter/software/pynauty/pynauty-$PYNAUTY.tar.gz | tar xz
  curl http://users.cecs.anu.edu.au/~bdm/nauty/nauty$NAUTY.tar.gz | tar xz

  mv nauty$NAUTY pynauty-$PYNAUTY/nauty
  mkdir -p "$HOME/.sources"
  mv pynauty-$PYNAUTY "$HOME/.sources/pynauty"
fi

make pynauty -C "$HOME/.sources/pynauty"
make user-ins -C "$HOME/.sources/pynauty"

# install all requirements
pip install -r requirements_test.txt
