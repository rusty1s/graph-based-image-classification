#!/bin/sh

pep8 **/*.py && \
nosetests test --with-coverage
