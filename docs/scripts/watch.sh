#!/bin/bash
SCRIPT_DIR=$(dirname $0)
ROOT_DIR=`realpath $SCRIPT_DIR/../..`
pushd $ROOT_DIR
    sphinx-autobuild docs docs/_build \
    --open-browser \
    --watch summer
popd
