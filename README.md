# A TensorFlow implementation of Graph-based Image Classification [![Build Status](https://travis-ci.org/rusty1s/graph-based-image-classification.png?branch=master)](https://travis-ci.org/rusty1s/graph-based-image-classification)

This is a TensorFlow implementation of the [Graph-based Convolutional Neural Network for Image Classification](https://github.com/rusty1s/deep-learning/tree/master/thesis).

## Why? [![start with why](https://img.shields.io/badge/start%20with-why%3F-brightgreen.svg?style=flat)](http://www.ted.com/talks/simon_sinek_how_great_leaders_inspire_action)

## Requirements

TensorFlow needs to be installed before running the training script.

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
