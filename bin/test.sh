#!/bin/sh

pep8 ./**/*.py --exclude=.sources && \
nosetests --with-coverage
