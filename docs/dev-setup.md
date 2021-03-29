# Developer Setup

 This document describes how to set up a local environemnt so that you can contribute to this project as a software developer. You do not need these instructions if you simply want to install and use this library.

 ## Install Miniconda

 Install the minimal Anaconda distribution, "Miniconda" [here](https://docs.conda.io/en/latest/miniconda.html)

## Install Poetry

Install the Poetry package manager [here](https://python-poetry.org/docs/#installation).

## Setup and configure conda envrionment

By this stage you should be able to run `conda` and `poetry` from the terminal. Once that is done:

```bash
# Create a new Python 3.6 environment with conda
conda create -n summer python=3.6
conda activate summer

# Find the conda env location for summer
conda env list
# Example output
# So in the case below, we have our summer env installed at "C:\tools\Anaconda3\envs\summer"
#
# conda environments:
#
# base                     C:\tools\Anaconda3
# autumn                   C:\tools\Anaconda3\envs\autumn
# summer                *  C:\tools\Anaconda3\envs\summer
#

# Configure Poetry to use our conda environment
poetry config settings.virtualenvs.path  C:/tools/Anaconda3/envs/summer

# Install packages with Poetry
poetry install

# Run unit tests to check
pytest -vv
```