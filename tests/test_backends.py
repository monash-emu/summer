import numpy as np

from computegraph.jaxify import has_jax
from summer.solver import SolverType

from .model_setup import get_test_model


def _test_compare_default_vectorized_outputs():
    """Simple direct-comparison acceptance test to check the vectorized backend produces
    outputs equivalent to the reference implementation
    """
    model = get_test_model(times=[0, 5])

    model.run(backend="reference", solver=SolverType.RUNGE_KUTTA)
    default_outputs = model.outputs.copy()

    model = get_test_model(times=[0, 5])
    model.run(backend="vectorized", solver=SolverType.RUNGE_KUTTA)
    vectorized_outputs = model.outputs

    np.testing.assert_allclose(vectorized_outputs, default_outputs)


def test_compare_default_jax_outputs():
    """Simple direct-comparison acceptance test to check the Jax backend produces
    outputs equivalent to the reference implementation
    """
    if not has_jax():
        return

    model = get_test_model(times=[0, 5])

    model.run(backend="vectorized", solver=SolverType.RUNGE_KUTTA, step_size=1.0)
    default_outputs = model.outputs.copy()

    model = get_test_model(times=[0, 5])
    model.run(backend="jax", solver=SolverType.RUNGE_KUTTA, parameters={})
    jax_outputs = model.outputs

    np.testing.assert_allclose(jax_outputs, default_outputs)
