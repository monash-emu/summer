# Developer Setup

This document describes how to set up Summer on your computer so that you can:

- contribute to this project as a software developer; or
- run the example notebooks listed [here](http://summerepi.com/examples/index.html)

You do not need these instructions if you simply want to install and use this library in your own project. Installation instructions for end-users can be found [here](http://summerepi.com/install.html#Installation-and-Quickstart).

## Download the Code

You will need to download the Summer codebase onto your computer using [Git](https://git-scm.com/)

```bash
# Clone the summer codebase into a folder named "summer".
git clone https://github.com/monash-emu/summer.git
```

## Install Miniconda

Install the minimal Anaconda distribution, "Miniconda" [here](https://docs.conda.io/en/latest/miniconda.html)

## Install Poetry

Install the Poetry package manager [here](https://python-poetry.org/docs/#installation).

## Setup and configure your conda envrionment

By this stage you should be able to run `conda` and `poetry` from the terminal. Once that is done:

```bash
# Step #1: make sure you are in the top-level directory of the summer repository
# Eg. C:\Users\matt\Documents\code\summer

# Step #2: create a new Python 3.7 environment with conda
conda create -n summer python=3.7
conda activate summer

# Find the conda env location for summer
conda env list
# See below for example output
# In this case, we have our summer env installed at "C:\tools\Anaconda3\envs\summer"
#
# conda environments:
#
# base                     C:\tools\Anaconda3
# autumn                   C:\tools\Anaconda3\envs\autumn
# summer                *  C:\tools\Anaconda3\envs\summer
#

# Configure Poetry to use our conda environment
poetry config virtualenvs.path  C:/tools/Anaconda3/envs/summer

# Install all required packages with Poetry
poetry install

# Run unit tests to check that everything is working
pytest -vv
```

Once this is done, you will be able to run the code examples in the documentation, instructions [here](https://github.com/monash-emu/summer#documentation).
