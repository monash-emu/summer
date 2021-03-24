import numpy as np
import pytest
from numpy.testing import assert_array_equal


from summer import stochastic

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
        #   [0.9996, 0.992, 1]
        # Leave prob of
        #   [4e-4, 8e-3, 0]
        # Giving us flow probabilities of approx
        #   [1e-4, 3e-3, 0], [3e-4, 5e-3, 0], [0.9996, 0.992, 1]
        # Which should sample to (using mock multinomial)
        #   [1, 3, 0], [3, 5, 0], [9996, 992, 1]
        # Which should be mapped to these changes in compartment size
        np.array([-4, 2, 3]),  # expected compartment size changes
    ]
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
    pass


def test_solve_stochastic():
    """
    - mock out get rates
    - mock out sample_entry_flows
    - mock out sample_transistion_flows
    """
    pass


def _mock_multinomial(size, probs):
    return np.round(size * probs)


def _mock_poisson(lam):
    # Fake poisson distribution
    return lam + 1
