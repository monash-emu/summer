import numpy as np
import pytest
from summer import CompartmentalModel, Multiply, Overwrite, Stratification

RANDOM_SEED = 1337


@pytest.mark.benchmark
def test_benchmark_default_ode_solver(benchmark):
    """
    Performance benchmark: check how long our models take to run.
    See: https://pytest-benchmark.readthedocs.io/en/stable/
    Run these with pytest -vv -m benchmark --benchmark-json benchmark.json
    """

    def run_default_ode_solver_test_model():
        model = _get_test_model()
        model.run()

    run_default_ode_solver_test_model()
    benchmark(run_default_ode_solver_test_model)


@pytest.mark.benchmark
def test_benchmark_rk4_ode_solver(benchmark):
    def run_rk4_solver_test_model():
        model = _get_test_model()
        model.run("rk4", step_size=0.1)

    benchmark(run_rk4_solver_test_model)


@pytest.mark.benchmark
def test_benchmark_stochastic_solver(benchmark):
    def run_stochastic_solver_test_model():
        model = _get_test_model(timestep=0.1)
        model.run_stochastic(RANDOM_SEED)

    benchmark(run_stochastic_solver_test_model)


def _get_test_model(timestep=1):
    comps = ["S", "EE", "LE", "EA", "LA", "R"]
    infectious_comps = ["LE", "EA", "LA"]
    model = CompartmentalModel(
        times=[0, 150],
        compartments=comps,
        infectious_compartments=infectious_comps,
        timestep=timestep,
    )
    model.set_initial_population({"S": int(20e6), "LA": 100})

    # Add flows
    model.add_infection_frequency_flow(name="infection", contact_rate=0.03, source="S", dest="EE")
    model.add_sojourn_flow(name="infect_onset", sojourn_time=7, source="EE", dest="LE")
    model.add_sojourn_flow(name="incidence", sojourn_time=7, source="LE", dest="EA")
    model.add_sojourn_flow(name="progress", sojourn_time=7, source="EA", dest="LA")
    model.add_sojourn_flow(name="recovery", sojourn_time=7, source="LA", dest="R")
    model.add_death_flow(name="infect_death", death_rate=0.005, source="LA")
    model.add_sojourn_flow(name="warning_immunity", sojourn_time=100, source="R", dest="S")

    # Stratify by age
    age_strat = Stratification("age", AGE_STRATA, comps)
    age_strat.set_population_split(AGE_SPLIT_PROPORTIONS)
    age_strat.set_mixing_matrix(AGE_MIXING_MATRIX)
    age_strat.add_flow_adjustments(
        "infection", {s: Multiply(v) for s, v in AGE_SUSCEPTIBILITY.items()}
    )
    model.stratify_with(age_strat)

    # Stratify by clinical status
    clinical_strat = Stratification("clinical", CLINICAL_STRATA, infectious_comps)
    clinical_strat.add_infectiousness_adjustments("LE", {**ADJ_BASE, "non_sympt": Overwrite(0.25)})
    clinical_strat.add_infectiousness_adjustments("EA", {**ADJ_BASE, "non_sympt": Overwrite(0.25)})
    clinical_strat.add_infectiousness_adjustments(
        "LA",
        {
            **ADJ_BASE,
            "non_sympt": Overwrite(0.25),
            "sympt_isolate": Overwrite(0.2),
            "hospital": Overwrite(0.2),
            "icu": Overwrite(0.2),
        },
    )
    clinical_strat.add_flow_adjustments(
        "infect_onset",
        {
            "non_sympt": Multiply(0.26),
            "icu": Multiply(0.01),
            "hospital": Multiply(0.04),
            "sympt_public": Multiply(0.66),
            "sympt_isolate": Multiply(0.03),
        },
    )
    model.stratify_with(clinical_strat)

    # Request derived outputs.
    model.request_output_for_flow(name="incidence", flow_name="incidence")
    model.request_output_for_flow(name="progress", flow_name="progress")
    for age in AGE_STRATA:
        for clinical in NOTIFICATION_STRATA:
            model.request_output_for_flow(
                name=f"progressXage_{age}Xclinical_{clinical}",
                flow_name="progress",
                dest_strata={"age": age, "clinical": clinical},
            )

    hospital_sources = []
    icu_sources = []
    for age in AGE_STRATA:
        icu_sources.append(f"progressXage_{age}Xclinical_icu")
        hospital_sources += [
            f"progressXage_{age}Xclinical_icu",
            f"progressXage_{age}Xclinical_hospital",
        ]

    model.request_aggregate_output(
        name="new_hospital_admissions",
        sources=hospital_sources,
    )
    model.request_aggregate_output(name="new_icu_admissions", sources=icu_sources)

    # Get notifications, which may included people detected in-country as they progress, or imported cases which are detected.
    notification_sources = [
        f"progressXage_{a}Xclinical_{c}" for a in AGE_STRATA for c in NOTIFICATION_STRATA
    ]
    model.request_aggregate_output(name="notifications", sources=notification_sources)

    # Infection deaths.
    model.request_output_for_flow(name="infection_deaths", flow_name="infect_death")
    model.request_cumulative_output(name="accum_deaths", source="infection_deaths")

    # Track hospital occupancy.
    # We count all ICU and hospital late active compartments and a proportion of early active ICU cases.
    model.request_output_for_compartments(
        "_late_active_hospital",
        compartments=["LA"],
        strata={"clinical": "hospital"},
        save_results=False,
    )
    model.request_output_for_compartments(
        "icu_occupancy",
        compartments=["LA"],
        strata={"clinical": "icu"},
    )
    model.request_output_for_compartments(
        "_early_active_icu",
        compartments=["EA"],
        strata={"clinical": "icu"},
        save_results=False,
    )
    proportion_icu_patients_in_hospital = 0.25
    model.request_function_output(
        name="_early_active_icu_proportion",
        func=lambda patients: patients * proportion_icu_patients_in_hospital,
        sources=["_early_active_icu"],
        save_results=False,
    )
    model.request_aggregate_output(
        name="hospital_occupancy",
        sources=[
            "_late_active_hospital",
            "icu_occupancy",
            "_early_active_icu_proportion",
        ],
    )

    # Proportion seropositive
    model.request_output_for_compartments(
        name="_total_population", compartments=comps, save_results=False
    )
    model.request_output_for_compartments(name="_recovered", compartments=["R"], save_results=False)
    model.request_function_output(
        name="proportion_seropositive",
        sources=["_recovered", "_total_population"],
        func=lambda recovered, total: recovered / total,
    )

    return model


CLINICAL_STRATA = ["non_sympt", "sympt_public", "sympt_isolate", "hospital", "icu"]
NOTIFICATION_STRATA = ["sympt_isolate", "hospital", "icu"]

ADJ_BASE = {
    "non_sympt": None,
    "sympt_public": None,
    "sympt_isolate": None,
    "hospital": None,
    "icu": None,
}
AGE_STRATA = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
    "75",
]

AGE_SUSCEPTIBILITY = {
    "0": 0.36,
    "5": 0.36,
    "10": 0.36,
    "15": 1.0,
    "20": 1.0,
    "25": 1.0,
    "30": 1.0,
    "35": 1.0,
    "40": 1.0,
    "45": 1.0,
    "50": 1.0,
    "55": 1.0,
    "60": 1.0,
    "65": 1.41,
    "70": 1.41,
    "75": 1.41,
}

AGE_SPLIT_PROPORTIONS = {
    "0": 0.08141605639918607,
    "5": 0.0774911659609209,
    "10": 0.07557396884304421,
    "15": 0.08248600637971966,
    "20": 0.08845878664211786,
    "25": 0.0898260548052294,
    "30": 0.09051616172885626,
    "35": 0.08408737823912063,
    "40": 0.06474992475900404,
    "45": 0.05854396769894234,
    "50": 0.05173561922188776,
    "55": 0.045466293192433206,
    "60": 0.03781177895976577,
    "65": 0.02865809888951674,
    "70": 0.020326268934260303,
    "75": 0.02285246934599485,
}


AGE_MIXING_MATRIX = np.array(
    [
        [
            3.55252067,
            1.50799267,
            0.72573171,
            0.4678699,
            0.72093328,
            1.17986202,
            1.32168824,
            1.03325714,
            0.56394422,
            0.33673388,
            0.33759545,
            0.26509214,
            0.16153679,
            0.10734965,
            0.05189907,
            0.02715479,
        ],
        [
            1.44724631,
            7.01133349,
            1.77923293,
            0.53702046,
            0.35880482,
            0.87228929,
            1.15664638,
            1.13558975,
            0.8771649,
            0.38450112,
            0.24691328,
            0.21507537,
            0.16409761,
            0.10113045,
            0.04102428,
            0.02776329,
        ],
        [
            0.49048181,
            2.61395704,
            9.66586805,
            1.34218585,
            0.62646942,
            0.56593748,
            0.73420226,
            0.96565435,
            1.06007971,
            0.58788931,
            0.30718083,
            0.15518553,
            0.09459451,
            0.08607218,
            0.05331695,
            0.03518695,
        ],
        [
            0.28781933,
            0.70858709,
            3.71911649,
            11.45024373,
            2.3529462,
            1.15007141,
            0.779031,
            0.9623547,
            1.09698541,
            0.98101948,
            0.49065853,
            0.19936554,
            0.12284964,
            0.05631184,
            0.02684475,
            0.01715252,
        ],
        [
            0.51214059,
            0.42740314,
            0.57995433,
            3.97698193,
            6.21379945,
            2.76453339,
            1.5796348,
            1.22960903,
            1.03372349,
            1.16489078,
            0.74586697,
            0.38087496,
            0.22997358,
            0.04568614,
            0.03891063,
            0.02772693,
        ],
        [
            1.01228315,
            0.59695992,
            0.33435457,
            1.27704623,
            3.25121679,
            4.56741323,
            2.32227448,
            1.59257876,
            1.25487035,
            1.01816371,
            0.89678369,
            0.45437165,
            0.31572985,
            0.05833624,
            0.02155394,
            0.01741524,
        ],
        [
            0.96825889,
            1.34650817,
            1.02316105,
            0.64124576,
            1.39861765,
            2.28260177,
            2.93787023,
            1.91686216,
            1.37353595,
            1.01068438,
            0.71581518,
            0.46450461,
            0.33977465,
            0.07147085,
            0.03222714,
            0.0225218,
        ],
        [
            0.83007301,
            1.30583079,
            1.10258494,
            0.83039396,
            0.8708053,
            1.55293371,
            1.79099851,
            2.44038256,
            1.75446371,
            1.07956352,
            0.69366747,
            0.34616377,
            0.28653132,
            0.10523939,
            0.04901417,
            0.01636075,
        ],
        [
            0.50633089,
            0.94037896,
            1.17120812,
            1.20069565,
            1.01717799,
            1.26900813,
            1.52433894,
            1.60716933,
            2.0789582,
            1.31186462,
            0.81573314,
            0.27327937,
            0.28540932,
            0.07970069,
            0.04328161,
            0.01974486,
        ],
        [
            0.4304938,
            0.67491607,
            0.81007979,
            1.46242312,
            0.9382396,
            1.01272342,
            1.13074941,
            1.20789486,
            1.23821176,
            1.40427156,
            0.7895627,
            0.32650166,
            0.21430459,
            0.04949065,
            0.03795077,
            0.03608425,
        ],
        [
            0.36980766,
            0.67349119,
            0.93112457,
            1.1541031,
            0.99132576,
            1.23532216,
            1.03089707,
            0.92432735,
            1.17372609,
            1.22355349,
            1.05962622,
            0.51538004,
            0.26719233,
            0.05370958,
            0.03490427,
            0.03427322,
        ],
        [
            0.62537642,
            0.76725476,
            0.62497741,
            0.78367375,
            0.69000786,
            1.07852766,
            1.02689581,
            0.72839779,
            0.75462314,
            0.63631753,
            0.69820759,
            0.68121041,
            0.36675839,
            0.10160119,
            0.03605388,
            0.02982956,
        ],
        [
            0.52836011,
            0.52427362,
            0.40467264,
            0.48780392,
            0.59120791,
            0.91369516,
            0.98100659,
            1.02413655,
            0.87085586,
            0.76006755,
            0.60272337,
            0.56721465,
            0.55438724,
            0.18209796,
            0.06451603,
            0.01997525,
        ],
        [
            0.29409262,
            0.4625957,
            0.3794538,
            0.20394258,
            0.2419993,
            0.31654444,
            0.4278979,
            0.41967292,
            0.33449649,
            0.17790234,
            0.16954871,
            0.20851669,
            0.20760425,
            0.30102474,
            0.09133641,
            0.0268338,
        ],
        [
            0.11806546,
            0.35446212,
            0.32212165,
            0.29319751,
            0.11757603,
            0.20623221,
            0.18573339,
            0.31410063,
            0.31832046,
            0.25661548,
            0.18540037,
            0.13063484,
            0.1793014,
            0.16340982,
            0.19038571,
            0.09398124,
        ],
        [
            0.16954409,
            0.24407132,
            0.35669035,
            0.30461366,
            0.09776579,
            0.10810445,
            0.14459499,
            0.22324768,
            0.22459548,
            0.22890467,
            0.20247261,
            0.1085315,
            0.05882404,
            0.09674404,
            0.07452203,
            0.08539959,
        ],
    ]
)
