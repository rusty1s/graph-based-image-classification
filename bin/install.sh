#!/bin/bash

usage="Graph-based Image Classification Install Script
-----------------------------------------------
Usage: $(basename "$0") [options...] <name>
Installs all requirements in a new conda environment with name <name>. Miniconda needs to be installed.

Options:
 -h, --help        This help text.
 -p, --python      The python version to use. (Default: 3.5)
     --tensorflow  The TensorFlow version to install. (Default: 0.11.0)
     --nauty       The nauty version to install. (Default: 26r7)
     --pynauty     The pynauty version to install. (Default: 0.6.0)"

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--help)
      echo "$usage"
      exit
      ;;
    -n|--name)
      NAME="$2"
      shift
      ;;
    -p|--python)
      PYTHON="$2"
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
      NAME="$1"
      ;;
  esac

  shift
done

# set default values
PYTHON="${PYTHON:-3.5}"
TENSORFLOW="${TENSORFLOW:-0.11.0}"
NAUTY="${NAUTY:-26r7}"
PYNAUTY="${PYNAUTY:-0.6.0}"

if [[ -z "$NAME" ]]; then
  echo "Abort: No <name> found. See --help for usage information."
  exit 1
fi

if ! hash conda 2>/dev/null; then
  echo "Abort: Miniconda is not installed on your system."
  exit 1
fi

# create conda environement
conda create -q -n "$NAME" python="$PYTHON"

# shellcheck disable=SC1091
source activate "$NAME"

# install conda requirements
conda install numpy scipy matplotlib
conda install -c https://conda.binstar.org/menpo opencv3

# install codecov for code coverage
pip install codecov

# install TensorFlow
if [[ $PYTHON == "2.7" ]]; then
  pip install "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp27-none-linux_x86_64.whl"
elif [[ $PYTHON == "3.4" ]]; then
  pip install "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp34-cp34m-linux_x86_64.whl"
elif [[ $PYTHON == "3.5" ]]; then
  pip install "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW-cp35-cp35m-linux_x86_64.whl"
fi

# install nauty and its python wrapper pynauty
if [[ ! -d "$HOME/.sources/pynauty" ]]; then
  curl "https://web.cs.dal.ca/~peter/software/pynauty/pynauty-$PYNAUTY.tar.gz" | tar xz
  curl "http://users.cecs.anu.edu.au/~bdm/nauty/nauty$NAUTY.tar.gz" | tar xz

  mv "nauty$NAUTY" "pynauty-$PYNAUTY/nauty"
  mkdir -p "$HOME/.sources"
  mv "pynauty-$PYNAUTY" "$HOME/.sources/pynauty"
fi

make pynauty -C "$HOME/.sources/pynauty"
make user-ins -C "$HOME/.sources/pynauty"

# install all requirements
pip install -r requirements_test.txt
