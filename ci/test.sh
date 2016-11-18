#!/bin/sh

# pep8 **/*.py --exclude=.sources && \
nosetests test --with-coverage
