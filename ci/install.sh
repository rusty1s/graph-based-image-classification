#!/bin/sh

source ci/conda.sh -v $TRAVIS_PYTHON_VERSION -n test --tensorflow $TENSORFLOW --nauty $NAUTY --pynauty $PYNAUTY
