# Summer: compartmental disease modelling in Python

[![Automated Tests](https://github.com/monash-emu/summer/actions/workflows/tests.yml/badge.svg)](https://github.com/monash-emu/summer/actions/workflows/tests.yml)

Summer is a compartmental disease modelling framework, written in Python. It provides a high-level API to build and run models. Features include:

- A variety of inter-compartmental flows (infections, sojourn, fractional, births, deaths, imports)
- Force of infection multipliers (frequency, density)
- Post-processing of compartment sizes into derived outputs
- Stratification of compartments, including:
  - Adjustments to flow rates based on strata
  - Adjustments to infectiousness based on strata
  - Heterogeneous mixing between strata
  - Multiple disease strains

Some helpful links:

- **[Documentation here](http://summerepi.com/)** with [code examples](http://summerepi.com/examples)
- [Available on PyPi](https://pypi.org/project/summerepi/) as `summerepi`.
- [Performance benchmarks](https://monash-emu.github.io/summer/)

## Installation and Quickstart

This project is tested with Python 3.6.
Install the `summerepi` package from PyPI

```bash
pip install summerepi
```

Then you can use the library to build and run models. See [here](http://summerepi.com/examples) for some code examples.

## Performance Note

You will find a significant performance improvement in the ODE solver if you set `OMP_NUM_THREADS` before importing `summer` or `numpy`.

```python
# Set this in your Python script
os.environ["OMP_NUM_THREADS"] = "1"

# Do it before importing summer or numpy
import summer
# ...
```

## Development

[Poetry](https://python-poetry.org/) is used for packaging and dependency management.

Initial project setup is documented [here](./docs/dev-setup.md) and should work for Windows or Ubuntu, maybe for MacOS.

Some common things to do as a developer working on this codebase:

```bash
# Activate summer conda environment prior to doing other stuff (see setup docs)
conda activate summer

# Install latest requirements
poetry install

# Publish to PyPI - use your PyPI credentials
poetry publish --build

# Add a new package
poetry add

# Run tests
pytest -vv

# Format Python code
black .
isort . --profile black
```

## Releases

- 1.0.X: Initial release

## Documentation

Sphinx is used to automatically build reference documentation for this library.
The documentation is automatically built and deployed to [summerepi.com](http://summerepi.com/) whenever code is pushed to `master`.

To build and deploy

```bash
./docs/scripts/build.sh
./docs/scripts/deploy.sh
```

To work on docs locally

```bash
# ./docs/scripts/watch.sh # FIXME - endless recursion
./docs/scripts/build.sh
# In a separate terminal
./docs/scripts/serve.sh
# Visit http://localhost:8000/
```
