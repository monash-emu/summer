# Summer: compartmental disease modelling in Python

[![Automated Tests](https://github.com/monash-emu/summer/actions/workflows/tests.yml/badge.svg)](https://github.com/monash-emu/summer/actions/workflows/tests.yml)

Summer is a Python-based framework for the creation and execution of [compartmental](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) (or "state-based") epidemiological models of infectious disease transmission.

It provides a range of structures for easily implementing compartmental models, including structure for some of the most common features added to basic compartmental frameworks, including:

- A variety of inter-compartmental flows (infections, transitions, births, deaths, imports)
- Force of infection multipliers (frequency, density)
- Post-processing of compartment sizes into derived outputs
- Stratification of compartments, including:
  - Adjustments to flow rates based on strata
  - Adjustments to infectiousness based on strata
  - Heterogeneous mixing between strata
  - Multiple disease strains

Some helpful links to learn more:

- [Rationale](http://summerepi.com/rationale.html) for why we are building Summer
- **[Documentation](http://summerepi.com/)** with [code examples](http://summerepi.com/examples)
- [Available on PyPi](https://pypi.org/project/summerepi/) as `summerepi`.
- [Performance benchmarks](https://monash-emu.github.io/summer/)

## Installation and Quickstart

This project requires at least Python 3.7 (and is actively targeted at 3.9)
Install the `summerepi` package from PyPI

```bash
pip install summerepi
```

Then you can use the library to build and run models. See [here](http://summerepi.com/examples) for some code examples.

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

Releases are numbered using [Semantic Versioning](https://semver.org/)

- 1.0.0/1:
  - Initial release
- 1.1.0:
  - Add stochastic integrator
- 2.0.2:
  - Rename fractional flow to transition flow
  - Remove sojourn flow
  - Add vectorized backend and other performance improvements
- 2.0.3:
  - Set default IVP solver to use a maximum step size of 1 timestep
- 2.0.4:
  - Add runtime derived values
- 2.0.5:
  - Remove legacy Summer implementation
- 2.1.0:
  - Add AdjustmentSystems
  - Improve vectorization of flows
  - Add computed_values inputs to flow and adjustment parameters
- 2.1.1:
  - Fix for invalid/unused package imports (cachetools)
- 2.2.0
  - Add validation and compartment caching optimizations
- 2.2.1
  - Derived output index caching
  - Optimized fast-tracks for infectious multipliers
- 2.2.2
  - JIT infectiousness calculations
  - Various micro-optimizations
- 2.2.3
  - Bugfix release (clamp outputs to 0.0)
- 2.2.4
  - Datetime awareness, DataFrame outputs
- 2.2.5
  - Performance improvements (frozenset), no API changes
- 2.2.6
  - Verify strata in flow adjustments (prevent unexpected behaviour)
- 2.2.7
  - Rename add_flow_adjustments -> set_flow_adjustments
- 2.2.8
  - Split imports functionality (add_importation_flow now requires split_imports arg)
- 2.2.9
  - Post-stratification population restribution
- 2.3.0
  - First official version to support only Python 3.7
- 2.5.0
  - Support Python 3.9
- 2.6.0
  - Merge 3.9/master branches
- 2.7.0
  - Include Python 3.10 support and update requirements
- 3.6.0
  - Summer 'classic' end-of-line release

## Release process

To do a release:

- Commit any code changes and push them to GitHub
- Choose a new release number accoridng to [Semantic Versioning](https://semver.org/)
- Add a release note above
- Edit the `version` key in `pyproject.toml` to reflect the release number
- Publish the package to [PyPI](https://pypi.org/project/summerepi/) using Poetry, you will need a PyPI login and access to the project
- Commit the release changes and push them to GitHub (Use a commit message like "Release 1.1.0")
- Update `requirements.txt` in Autumn to use the new version of Summer

```bash
poetry build
poetry publish
```

## Documentation

Sphinx is used to automatically build reference documentation for this library.
The documentation is automatically built and deployed to [summerepi.com](http://summerepi.com/) whenever code is pushed to `master`.

To run or edit the code examples in the documentation, start a jupyter notebook server as follows:

```bash
jupyter notebook --config docs/jupyter_notebook_config.py
# Go to http://localhost:8888/tree/docs/examples in your web browser.
```

You can clean outputs from all the example notbooks with

```bash
./docs/scripts/clean.sh
```

To build and deploy

```bash
./docs/scripts/build.sh
./docs/scripts/deploy.sh
```

To work on docs locally

```bash
./docs/scripts/watch.sh
```
