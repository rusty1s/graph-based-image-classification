# A TensorFlow implementation of Graph-based Image Classification

[![Build Status](https://travis-ci.org/rusty1s/graph-based-image-classification.svg?branch=master)](https://travis-ci.org/rusty1s/graph-based-image-classification)
[![codecov.io Code Coverage](https://img.shields.io/codecov/c/github/rusty1s/graph-based-image-classification.svg?maxAge=2592000)](https://codecov.io/github/rusty1s/graph-based-image-classification?branch=master)

This is a TensorFlow implementation of the [Graph-based Image Classification](https://github.com/rusty1s/deep-learning/tree/master/thesis) master thesis.

## Requirements

Project is tested on Python 2.7, 3.4 and 3.5.

[TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation), [nauty](http://pallini.di.uniroma1.it/) and its python wrapper [pynauty](https://web.cs.dal.ca/~peter/software/pynauty/html/install.html) need to be installed before running the script.
See [`install.sh`](https://github.com/rusty1s/graph-based-image-classification/blob/master/ci/install.sh) for information on how to do so.

**Supported Versions:**
* TensorFlow: 0.11.0 and the current `master` version
* pynauty: 0.6.0

To install the additional required python packages, run:

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
