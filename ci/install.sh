#!/bin/bash

source bin/conda.sh
source bin/install.sh --python "$TRAVIS_PYTHON_VERSION" --tensorflow "$TENSORFLOW" --nauty "$NAUTY" --pynauty "$PYNAUTY" test
