import numpy as np

from .model_setup import _get_test_model

def test_compare_default_vectorized_outputs():
    """Simple direct-comparison smoke test to check the vectorized backend produces
    the same outputs as the default
    """
    model = _get_test_model(times=[0,5])

    model.run(backend='default')
    default_outputs = model.outputs.copy()

    model.run(backend='vectorized')
    vectorized_outputs = model.outputs

    assert((default_outputs == vectorized_outputs).all())
