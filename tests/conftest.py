# PyTest configuration file.
# See pytest fixtue docs: https://docs.pytest.org/en/latest/fixture.html
import os

# Ensure NumPy only uses 1 thread for matrix multiplication,
# because numpy is stupid and tries to use heaps of threads which is quite wasteful
# and it makes our models run way more slowly.
os.environ["OMP_NUM_THREADS"] = "1"

import pytest
from summer2.model import BackendType

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


def pytest_configure(config):
    config.addinivalue_line("markers", "github_only: Mark test to run only in GitHub Actions")
    config.addinivalue_line(
        "markers", "benchmark: A test which benchmarks the performance of some code"
    )


def pytest_runtest_setup(item):
    for _ in item.iter_markers(name="benchmark"):
        if not IS_GITHUB_CI:
            pytest.skip("Long running test: run on GitHub only.")

    for marker in item.iter_markers(name="github_only"):
        if not IS_GITHUB_CI:
            pytest.skip("Long running test: run on GitHub only.")


@pytest.fixture(params=[BackendType.PYTHON])
def backend(request):
    return request.param
