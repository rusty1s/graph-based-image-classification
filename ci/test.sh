#!/bin/sh

pep8 **/*.py && \
nosetests test
