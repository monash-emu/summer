#!/bin/bash
set -e
SCRIPT_DIR=$(dirname $0)
DOCS_DIR=`realpath $SCRIPT_DIR/..`

echo -e "\nCleaning notebooks"
pushd $DOCS_DIR
    example_notebooks=$(find examples -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*')
    for nb in "$example_notebooks"
    do
        jupyter nbconvert --clear-output --inplace $nb
    done

popd
echo -e "\nFinished cleaning notebooks"
