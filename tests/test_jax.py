import numpy as np

from summer import CompartmentalModel, Stratification, StrainStratification
from summer.solver import SolverType
from summer.runner.jax import build_model_with_jax
from summer.parameters import Parameter
from summer.adjust import Overwrite


from tests.test_params.models import PARAMS, build_model_params, build_model_mixing_func


def test_model_params():
    params = PARAMS["params"]
    m, jrun = build_model_with_jax(build_model_params)

    m.run(solver=SolverType.ODE_INT, parameters=params, rtol=1.4e-8, atol=1.4e-8)
    joutputs = jrun(params)

    np.testing.assert_allclose(joutputs, m.outputs, atol=1e-5)


def test_model_mm_func():
    params = PARAMS["params_mixing_func"]
    m, jrun = build_model_with_jax(build_model_mixing_func)

    m.run(solver=SolverType.ODE_INT, parameters=params, rtol=1.4e-8, atol=1.4e-8)
    joutputs = jrun(params)

    np.testing.assert_allclose(joutputs, m.outputs, atol=1e-5)


def test_model_multistrat_strains():
    parameters = {
        "age_split": {"young": 0.8, "old": 0.2},
        "contact_rate": 0.1,
        "strain_infect_adjust.wild_type": 1.1,
        "strain_infect_adjust.variant1": 0.9,
        "strain_infect_adjust.variant2": 1.3,
    }

    def get_ipop_dist(total, infected_prop):
        num_infected = total * infected_prop
        return {"S": total - num_infected, "I": num_infected, "R": 0}

    def build_model(**kwargs):
        model = CompartmentalModel((0, 100), ["S", "I", "R"], ["I"], takes_params=True)

        model.set_initial_population(get_ipop_dist(1000.0, 0.4))

        strat = Stratification("age", ["young", "old"], ["S", "I", "R"])
        strat.set_population_split(Parameter("age_split"))
        model.stratify_with(strat)

        model.add_infection_frequency_flow("infection", Parameter("contact_rate"), "S", "I")

        strain_strat = StrainStratification("strain", ["wild_type", "variant1", "variant2"], ["I"])

        strain_strat.add_infectiousness_adjustments(
            "I",
            {
                "wild_type": Parameter("strain_infect_adjust.wild_type"),
                "variant1": Overwrite(Parameter("strain_infect_adjust.variant1")),
                "variant2": Overwrite(Parameter("strain_infect_adjust.variant2")),
            },
        )

        model.stratify_with(strain_strat)

        model.add_death_flow("death_after_infection", 0.01, "I")

        return model

    m_nojax, jaxrun = build_model_with_jax(build_model)

    m_nojax.run(parameters=parameters)
    joutputs = jaxrun(parameters)

    np.testing.assert_allclose(joutputs, m_nojax.outputs, atol=1e-5)
