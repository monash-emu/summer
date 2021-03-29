import numpy as np
import pytest
from numpy.testing import assert_array_equal


from summer import stochastic, CompartmentalModel
from summer.flows import DeathFlow, FractionalFlow
from summer.compartment import Compartment

ENTRY_FLOW_TESTS = [
    # timestep, flow rates, expected
    (1, np.array([1.0, 2.0, 3.0]), np.array([2, 3, 4])),
    (0.1, np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.2, 1.3])),
]


@pytest.mark.parametrize("timestep, flow_rates, expected_arr", ENTRY_FLOW_TESTS)
def test_sample_entry_flows(monkeypatch, timestep, flow_rates, expected_arr):
    # Replace jitted version of func with original func
    monkeypatch.setattr(stochastic, "_sample_entry_flows", stochastic._sample_entry_flows.py_func)
    monkeypatch.setattr(stochastic.np.random, "poisson", _mock_poisson)
    entry_changes = stochastic.sample_entry_flows(None, flow_rates, timestep)
    assert_array_equal(entry_changes, expected_arr)


TRANSITION_FLOW_TESTS = [
    # 2 flows, 3 compartments
    # timestep, flow_rates, flow_map, comp_vals, expected
    [
        1,  # timestep
        np.array([[1.0, 0, 0], [0, 5.0, 0]]),  # flow rates
        np.array([[0, 0, 1], [1, 1, 2]]),  # flow map (flow idx, source idx, dest idx)
        np.array([10000, 1000, 100]),  # compartment values
        # We expect to get normalized flow rates of
        #   [1e-4, 0, 0], [0, 5e-3, 0]
        # Total flows of
        #   [1e-4, 5e-3, 0]
        # Flow proportions of
        #   [1, 0, 0], [0, 1, 0]
        # Stay prob of
        #   [0.9999, 0.995, 1]
        # Leave prob of about
        #   [1e-4, 5e-3, 0]
        # Giving us flow probabilities of approx
        #   [1e-4, 0, 0], [0, 5e-3, 0], [0.9999, 0.995, 1]
        # Which should sample to (using mock multinomial)
        #   [1, 0, 0], [0, 5, 0], [9999, 995, 1]
        # Which should be mapped to these changes in compartment size
        np.array([-1, -4, 5]),  # expected compartment size changes
    ]
    # TODO: Test some more exotic cases (different timesteps, higher/lower flow rates)
]


@pytest.mark.parametrize(
    "timestep, flow_rates, flow_map, comp_vals, expected_arr", TRANSITION_FLOW_TESTS
)
def test_sample_transistion_flows(
    monkeypatch, timestep, flow_rates, flow_map, comp_vals, expected_arr
):
    # Replace jitted version of func with original func
    monkeypatch.setattr(
        stochastic, "_sample_flows_multinomial", stochastic._sample_flows_multinomial.py_func
    )
    monkeypatch.setattr(stochastic.np.random, "multinomial", _mock_multinomial)
    trans_changes = stochastic.sample_transistion_flows(
        None, flow_rates, flow_map, comp_vals, timestep
    )
    assert_array_equal(trans_changes, expected_arr)


def test_build_flow_map():
    S = Compartment("S")
    S.idx = 0
    I = Compartment("I")
    I.idx = 1
    R = Compartment("R")
    R.idx = 2
    exit_a = DeathFlow("a", source=S, param=1.0)
    exit_b = DeathFlow("b", source=I, param=1.0)
    trans_c = FractionalFlow("c", source=S, dest=I, param=1.0)
    trans_d = FractionalFlow("d", source=I, dest=R, param=1.0)
    trans_e = FractionalFlow("e", source=R, dest=S, param=1.0)
    flows = [exit_a, exit_b, trans_c, trans_d, trans_e]
    expected_map = np.array(
        [
            [0, 0, stochastic.NO_DESTINATION],
            [1, 1, stochastic.NO_DESTINATION],
            [2, 0, 1],
            [3, 1, 2],
            [4, 2, 0],
        ]
    )
    actual_map = stochastic.build_flow_map(flows)
    assert_array_equal(expected_map, actual_map)


def test_solve_stochastic(monkeypatch):
    """
    Test that _solve_stochastic glue code works.
    Don't test the actual flow rate calculations or stochastic sampling bits.
    """
    # Mock out stochastic flow sampling - tested elsewhere.
    def mock_sample_entry_flows(seed, entry_flow_rates, timestep):
        return entry_flow_rates

    def mock_sample_transistion_flows(seed, flow_rates, flow_map, comp_vals, timestep):
        return flow_rates.sum(axis=0)

    monkeypatch.setattr(stochastic, "sample_entry_flows", mock_sample_entry_flows)
    monkeypatch.setattr(stochastic, "sample_transistion_flows", mock_sample_transistion_flows)

<<<<<<< HEAD
    # Mock out flow rate calculation - tested elsewhere and tricky to predict.
=======
>>>>>>> newcode
    model = CompartmentalModel(
        times=[0, 5],
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
    )
<<<<<<< HEAD

    def mock_get_rates(comp_vals, time):
        # return the flow rates that will be used to solve the model
        return None, np.array([8.0, 6.0, 3.0, 2.0])

    monkeypatch.setattr(model, "_get_rates", mock_get_rates)

=======
>>>>>>> newcode
    # Add some people to the model, expect initial conditions of [990, 10, 0]
    model.set_initial_population(distribution={"S": 990, "I": 10})
    # Add flows - the parameters add here will be overidden by  `mock_get_rates`
    # but the flow directions will be used.
    model.add_crude_birth_flow("birth", 8, "S")
    model.add_infection_frequency_flow("infection", 6, "S", "I")
    model.add_death_flow("infect_death", 3, "I")
    model.add_fractional_flow("recovery", 2, "I", "R")
<<<<<<< HEAD
=======

    # Mock out flow rate calculation - tested elsewhere and tricky to predict.
    def mock_get_rates(comp_vals, time):
        # return the flow rates that will be used to solve the model
        return None, np.array([float(f.param) for f in model._flows])

    monkeypatch.setattr(model, "_get_rates", mock_get_rates)

>>>>>>> newcode
    model.run_stochastic()
    expected_outputs = np.array(
        [
            [990, 10, 0],
            [992, 11, 2],
            [994, 12, 4],
            [996, 13, 6],
            [998, 14, 8],
            [1000, 15, 10],
        ]
    )
    assert_array_equal(model.outputs, expected_outputs)


def _mock_multinomial(size, probs):
    return np.round(size * probs)


def _mock_poisson(lam):
    # Fake poisson distribution
    return lam + 1
