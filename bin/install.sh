#!/bin/bash

usage="Graph-based Image Classification Install Script
-----------------------------------------------
Usage: $(basename "$0") [options...] <name>

Installs all requirements in a new conda environment with name <name>. Miniconda needs to be installed.

Options:
 -h, --help        This help text.
 -p, --python      The python version to use. (Default: 3.5)
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

# Set default values.
PYTHON="${PYTHON:-3.5}"
NAUTY="${NAUTY:-26r7}"
PYNAUTY="${PYNAUTY:-0.6.0}"

if [[ -z "$NAME" ]]; then
  echo "Abort: No <name> found. See --help for usage information."
  exit 1
fi

if ! hash conda 2>/dev/null; then
  echo "Abort: Miniconda is not installed on your system."
  echo "Run ./bin/conda.sh to install Miniconda."
  exit 1
fi

# Create conda environement.
conda create -q -n "$NAME" python="$PYTHON"

# shellcheck disable=SC1091
source activate "$NAME"

# Install conda packages.
conda install numpy scipy matplotlib

# Install nauty and its python wrapper pynauty.
if [[ ! -d "$HOME/.sources/pynauty" ]]; then
  curl "https://web.cs.dal.ca/~peter/software/pynauty/pynauty-$PYNAUTY.tar.gz" | tar xz
  curl "http://users.cecs.anu.edu.au/~bdm/nauty/nauty$NAUTY.tar.gz" | tar xz

  mv "nauty$NAUTY" "pynauty-$PYNAUTY/nauty"
  mkdir -p "$HOME/.sources"
  mv "pynauty-$PYNAUTY" "$HOME/.sources/pynauty"

  cp -f "__pynauty__/graph.py" "$HOME/.sources/pynauty/src/graph.py"
  cp -f "__pynauty__/nautywrap.c" "$HOME/.sources/pynauty/src/nautywrap.c"
fi

make pynauty -C "$HOME/.sources/pynauty"
make user-ins -C "$HOME/.sources/pynauty"

# Install all requirements.
pip install -r requirements_test.txt
