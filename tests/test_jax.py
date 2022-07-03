import numpy as np

from summer.solver import SolverType
from summer.runner.jax import build_model_with_jax

from tests.test_params.models import PARAMS, build_model_params, build_model_mixing_func


def test_model_params():
    params = PARAMS["params"]
    m, jrun = build_model_with_jax(build_model_params, params)

    m.run(solver=SolverType.ODE_INT, parameters=params, rtol=1.4e-8, atol=1.4e-8)
    joutputs = jrun(params)

    np.testing.assert_allclose(joutputs, m.outputs, atol=1e-5)


def test_model_mm_func():
    params = PARAMS["params_mm_func"]
    m, jrun = build_model_with_jax(build_model_mixing_func, params)

    m.run(solver=SolverType.ODE_INT, parameters=params, rtol=1.4e-8, atol=1.4e-8)
    joutputs = jrun(params)

    np.testing.assert_allclose(joutputs, m.outputs, atol=1e-5)
