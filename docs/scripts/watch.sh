#!/bin/bash
SCRIPT_DIR=$(dirname $0)
ROOT_DIR=`realpath $SCRIPT_DIR/../..`
# FIXME: Endless recursion
pushd $ROOT_DIR
    watchmedo shell-command \
        --command="./docs/scripts/build.sh" \
        --patterns="*.py;*.md;*.rst" \
        --ignore-patterns="./.venv/*;./docs/_build/*;./docs/examples/*" \
        --recursive --wait --drop \
        .
popd
