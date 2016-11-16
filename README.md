# A TensorFlow implementation of Graph-based Image Classification

[![Build Status](https://travis-ci.org/rusty1s/graph-based-image-classification.svg?branch=master)](https://travis-ci.org/rusty1s/graph-based-image-classification)
[![codecov.io Code Coverage](https://img.shields.io/codecov/c/github/rusty1s/graph-based-image-classification.svg?maxAge=2592000)](https://codecov.io/github/rusty1s/graph-based-image-classification?branch=master)

This is a TensorFlow implementation of the [Graph-based Image Classification](https://github.com/rusty1s/deep-learning/tree/master/thesis) master thesis.

## Requirements

Project is tested on Python 2.7, 3.4 and 3.5.
TensorFlow needs to be installed before running the script.
TensorFlow 0.11.0 and the current `master` version are supported.

To install the required python packages (except TensorFlow), run:

```bash
pip install -r requirements.txt
```

## Running tests

Install the test requirements:

```bash
pip install -r requirements_test.txt
```

Run the test suite:

```bash
./ci/test.sh
```
