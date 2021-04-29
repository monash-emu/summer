import numpy as np

from summer.solver import SolverType

from .model_setup import _get_test_model

def test_compare_default_vectorized_outputs():
    """Simple direct-comparison acceptance test to check the vectorized backend produces
    outputs equivalent to the reference implementation
    """
    model = _get_test_model(times=[0,5])

    model.run(backend='reference', solver=SolverType.RUNGE_KUTTA)
    default_outputs = model.outputs.copy()

    model.run(backend='vectorized', solver=SolverType.RUNGE_KUTTA)
    vectorized_outputs = model.outputs

    np.testing.assert_allclose(vectorized_outputs, default_outputs)
