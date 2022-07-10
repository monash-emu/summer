import numpy as np

from summer import CompartmentalModel
from summer.parameters.params import CompartmentValues, Function, Parameter

from tests.test_params.models import (
    PARAMS,
    BUILD_FUNCTIONS,
    build_model_static,
    build_model_static_func,
)


def test_param_static_equal():
    mkey = "params"
    build_model = BUILD_FUNCTIONS[mkey]
    parameters = PARAMS[mkey]

    m_param = build_model()
    m_param.run(parameters=parameters)

    m_static = build_model_static(parameters)

    m_static.run()

    assert (m_param.outputs == m_static.outputs).all()


def test_param_func_equal():
    mkey = "params_func"
    build_model = BUILD_FUNCTIONS[mkey]
    parameters = PARAMS[mkey]

    m_param = build_model()
    m_param.run(parameters=parameters)

    m_static = build_model_static_func(parameters)
    m_static.run()

    assert (m_param.outputs == m_static.outputs).all()


def test_param_model_strat():
    mkey = "params_strat"
    build_model = BUILD_FUNCTIONS[mkey]
    parameters = PARAMS[mkey]

    m_param = build_model()

    m_param.run(parameters=parameters)


def test_param_model_mixing_func():
    mkey = "params_mixing_func"
    build_model = BUILD_FUNCTIONS[mkey]
    parameters = PARAMS[mkey]

    m_param = build_model()
    m_param.run(parameters=parameters)


def test_param_model_derived_outputs():

    mkey = "params_new_derived"
    build_model = BUILD_FUNCTIONS[mkey]
    parameters = PARAMS[mkey]

    m_param = build_model()
    m_param.run(parameters=parameters)
    do_df = m_param.get_derived_outputs_df()

    assert set(do_df.columns) == set(
        ["total_population", "recovered", "proportion_seropositive", "prop_seropositive_surveyed"]
    )


def test_param_model_computed_values():
    mkey = "params_new_derived"
    build_model = BUILD_FUNCTIONS[mkey]
    parameters = PARAMS[mkey]

    m = build_model()

    def get_scaled_pop(compartment_values, scale):
        return compartment_values.sum() * scale

    m.add_computed_value_func(
        "scaled_pop", Function(get_scaled_pop, [CompartmentValues, Parameter("pop_scale")])
    )
    m.request_computed_value_output("scaled_pop")

    mm = np.array((0.1, 0.2, 0.1, 0.5)).reshape(2, 2)
    parameters = {
        "pop_scale": 0.5,
        "recovery_rate": 0.02,
        "contact_scale": 1.0,
        "aged_recovery_scale": 0.5,
        "mixing_matrix": mm,
        "young_infect_scale": 3.0,
        "matrix_scale": 0.5,
        "serosurvey_scale": 0.5,
    }

    m.run(parameters=parameters)

    do_df = m.get_derived_outputs_df()

    assert "scaled_pop" in do_df.columns

    np.testing.assert_allclose(do_df["scaled_pop"], 50.0)


def test_param_initial_pop():
    model = CompartmentalModel((0, 100), ["S", "I", "R"], ["I"])

    def get_ipop_dist(total, infected_prop):
        num_infected = total * infected_prop
        return {"S": total - num_infected, "I": num_infected, "R": 0}

    ipop_param = Parameter("initial_pop_distribution")

    model.set_initial_population(ipop_param)

    parameters = {"initial_pop_distribution": get_ipop_dist(1000.0, 0.4)}
    model.run(parameters=parameters)
    assert (model.outputs[0] == np.array((600, 400, 0))).all()
