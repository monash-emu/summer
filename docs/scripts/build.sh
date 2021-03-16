#!/bin/bash
set -e
SCRIPT_DIR=$(dirname $0)
DOCS_DIR=`realpath $SCRIPT_DIR/..`

echo -e "\nBuilding documentation HTML"
pushd $DOCS_DIR
    rm -rf ./_build/doctrees  ./_build/html/*
    make html
popd
echo -e "\nFinished building documentation HTML"
