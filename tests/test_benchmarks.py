import pytest

from .model_setup import get_test_model

RANDOM_SEED = 1337


@pytest.mark.benchmark
def test_benchmark_default_ode_solver(benchmark):
    """
    Performance benchmark: check how long our models take to run.
    See: https://pytest-benchmark.readthedocs.io/en/stable/
    Run these with pytest -vv -m benchmark --benchmark-json benchmark.json
    """

    def run_default_ode_solver_test_model():
        model = get_test_model()
        model.run()

    benchmark(run_default_ode_solver_test_model)


@pytest.mark.benchmark
def test_benchmark_rk4_ode_solver(benchmark):
    def run_rk4_solver_test_model():
        model = get_test_model()
        model.run("rk4", step_size=0.1)

    benchmark(run_rk4_solver_test_model)


@pytest.mark.benchmark
def test_benchmark_stochastic_solver(benchmark):
    def run_stochastic_solver_test_model():
        model = get_test_model(timestep=0.1)
        model.run_stochastic(RANDOM_SEED)

    benchmark(run_stochastic_solver_test_model)
