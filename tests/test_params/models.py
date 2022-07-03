import numpy as np

from summer import CompartmentalModel, Stratification

from summer.parameters.params import (
    Parameter,
    Time,
    Function,
    DerivedOutput,
    # Leave this in to appease flake8, but also as a reminder to implement tests using them!
    # ComputedValue,
    # find_all_parameters,
    # Variable,
    # model_var,
)

param = Parameter
func = Function

_mm = np.array((0.1, 0.2, 0.1, 0.5)).reshape(2, 2)


PARAMS = {
    "params": {"contact_rate": 1.0, "recovery_rate": 0.01},
    "params_func": {"recovery_rate": 0.02, "contact_scale": 1.0},
    "params_strat": {
        "recovery_rate": 0.02,
        "contact_scale": 5.0,
        "aged_recovery_scale": 0.5,
        "mixing_matrix": _mm,
        "young_infect_scale": 5.0,
    },
    "params_mm_func": {
        "recovery_rate": 0.02,
        "contact_scale": 1.0,
        "aged_recovery_scale": 0.5,
        "mixing_matrix": _mm,
        "young_infect_scale": 3.0,
        "matrix_scale": 0.5,
    },
    "params_new_derived": {
        "recovery_rate": 0.02,
        "contact_scale": 1.0,
        "aged_recovery_scale": 0.5,
        "mixing_matrix": _mm,
        "young_infect_scale": 3.0,
        "matrix_scale": 0.5,
        "serosurvey_scale": 0.5,
    },
}


def build_model_params(**kwargs) -> CompartmentalModel:
    """
    Base parameterized model
    """
    m = CompartmentalModel([0, 100], ["S", "I", "R"], ["I"])
    m.set_initial_population(dict(S=90, I=10, R=0))
    m.add_infection_frequency_flow("infection", param("contact_rate"), "S", "I")
    m.add_transition_flow("recovery", param("recovery_rate"), "I", "R")
    return m


def build_model_static(params, **kwargs):
    """
    Static model equivalent to base build_model_params
    """
    m = CompartmentalModel([0, 100], ["S", "I", "R"], ["I"])
    m.set_initial_population(dict(S=90, I=10, R=0))
    m.add_infection_frequency_flow("infection", params["contact_rate"], "S", "I")
    m.add_transition_flow("recovery", params["recovery_rate"], "I", "R")
    return m


def build_model_params_func(**kwargs):
    """Model with parameterized Function"""

    def custom_rate(time, scale):
        return (time * 0.1) * scale

    m = CompartmentalModel([0, 100], ["S", "I", "R"], ["I"])
    m.set_initial_population(dict(S=90, I=10, R=0))
    m.add_infection_frequency_flow(
        "infection",
        func(
            custom_rate,
            args=(
                Time,
                param("contact_scale"),
            ),
        ),
        "S",
        "I",
    )
    m.add_transition_flow("recovery", param("recovery_rate"), "I", "R")
    return m


def build_model_static_func(params, **kwargs):
    """Static equivalent to build_model_params_func"""

    def custom_rate(time, cv):
        return (time * 0.1) * params["contact_scale"]

    m = CompartmentalModel([0, 100], ["S", "I", "R"], ["I"])
    m.set_initial_population(dict(S=90, I=10, R=0))
    m.add_infection_frequency_flow("infection", custom_rate, "S", "I")
    m.add_transition_flow("recovery", params["recovery_rate"], "I", "R")
    return m


def build_model_params_strat(**kwargs):
    """Stratified parameterized model"""
    m = build_model_params_func()
    age_strat = Stratification(name="age", strata=["young", "old"], compartments=["S", "I", "R"])

    def constant(value):
        return value

    # Infectiousness adjustment parameter
    age_strat.add_infectiousness_adjustments(
        "I", {"old": None, "young": param("young_infect_scale")}
    )

    # Include static mixing matrix as parameter
    age_strat.set_mixing_matrix(param("mixing_matrix"))

    # Flow adjustment ModelFunction
    age_strat.set_flow_adjustments(
        "recovery",
        {"young": None, "old": func(constant, kwargs={"value": param("aged_recovery_scale")})},
    )

    m.stratify_with(age_strat)

    return m


def build_model_mixing_func(use_jax=False):
    """Stratified model with a custom fucntion for mixing matrix"""
    # Use our base parameterized model as above
    m = build_model_params_func()
    age_strat = Stratification(name="age", strata=["young", "old"], compartments=["S", "I", "R"])

    # Infectiousness adjustment parameter
    age_strat.add_infectiousness_adjustments(
        "I", {"old": None, "young": param("young_infect_scale")}
    )

    if use_jax:
        from jax import numpy as fnp
    else:
        import numpy as fnp

    def scale_and_clamp(matrix, scale):
        mm = matrix * scale
        return fnp.clip(mm, 0.0, 1.0)

    # Include static mixing matrix as parameter
    age_strat.set_mixing_matrix(
        func(scale_and_clamp, [param("mixing_matrix"), param("matrix_scale")])
    )

    # Flow adjustment ModelFunction
    age_strat.set_flow_adjustments("recovery", {"young": None, "old": param("aged_recovery_scale")})

    m.stratify_with(age_strat)

    return m


def build_model_new_derived(**kwargs):
    """Model with new style derived outputs"""
    m = build_model_mixing_func()
    m.request_output_for_compartments("total_population", ["S", "I", "R"])
    m.request_output_for_compartments("recovered", ["R"])

    def calc_prop(numerator, denominator):
        return numerator / denominator

    m.request_param_function_output(
        "proportion_seropositive",
        func(calc_prop, [DerivedOutput("recovered"), DerivedOutput("total_population")]),
    )

    def scale(value, scale):
        return value * scale

    m.request_param_function_output(
        "prop_seropositive_surveyed",
        func(
            scale,
            kwargs={
                "value": DerivedOutput("proportion_seropositive"),
                "scale": param("serosurvey_scale"),
            },
        ),
    )

    return m
