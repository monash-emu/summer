import pytest

import numpy as np
from scipy.stats import binom

from summer import CompartmentalModel

TRIALS = 1000
ERROR_RATE = 1e-5  # we expect this test to fail 1/100000 times


DEATH_EXTINCTION_PARAMS = [
    # infect death rate, contact rate
    (0.5, 2),
    (0.1, 1),
    (0.9, 3),
]


@pytest.mark.parametrize("death_rate, contact_rate", DEATH_EXTINCTION_PARAMS)
def test_stochastic_death_exitinction(death_rate, contact_rate):
    """
    A smokey test to make sure the disease goes extinct around the right amount,
    because the infectious person dies before they can infect someone else.

    See here for how this stuff is calculated
    https://autumn-files.s3-ap-southeast-2.amazonaws.com/Switching_to_stochastic_mode.pdf

    Consider the following flow rates:
    - 0.5 infected deaths timestep
    - 2 people infected per timestep
        - infection frequency force of infection of  1 inf / 1000 pop
        - sus pop of 999
        - contact rate of 2
        - flow rate of 2 * 999 / 1000 = 1.998 ~= 2

    Based on stochastic model (per person)
    - P(infect_death) ~=40%(1 - e^(-0.5/1))
    - P(infected) ~= 0.2% (1 - e^(-2/1000))

    Using a binomial calculator, we get
    - ~86% chance of 1 or more people getting infected
    - ~14% chance of noone getting infected

    Death and infection are independent processes within the model.
    So then we expect a ~6% chance of exctinction (infected person dies, no one infected) (40% * 14%)

    Given this there is a > 0.999999 chance that we see at least 25
    disease exctinctions in 1000 runs (using binomial calculation)
    """
    pr_death = 1 - np.exp(-death_rate)
    pr_infected = 1 - np.exp(-contact_rate / 1000)
    pr_noone_infected = binom.pmf(0, 1000, pr_infected)
    pr_extinction = pr_death * pr_noone_infected
    expected_extinctions = _find_num_successes(pr_extinction, TRIALS, ERROR_RATE)
    count_extinctions = 0
    for _ in range(TRIALS):
        model = CompartmentalModel(
            times=[0, 1],
            compartments=["S", "I", "R"],
            infectious_compartments=["I"],
        )
        model.set_initial_population(distribution={"S": 999, "I": 1})
        model.add_death_flow("infect_death", death_rate, "I")
        model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
        model.run_stochastic()
        is_extinct = model.outputs[1, 1] == 0
        if is_extinct:
            count_extinctions += 1

    assert count_extinctions >= expected_extinctions


RECOVERY_EXTINCTION_PARAMS = [
    # recovery rate, contact rate
    (0.5, 2),
    (0.2, 2),
    (0.1, 1),
]


@pytest.mark.parametrize("recovery_rate, contact_rate", RECOVERY_EXTINCTION_PARAMS)
def test_stochastic_recovery_exitinction(recovery_rate, contact_rate):
    """
    A smokey test to make sure the disease goes extinct sometimes,
    because the infectious person recovers before they can infect someone else.

    Calculations similar to test_stochastic_death_exitinction
    """
    pr_recovery = 1 - np.exp(-recovery_rate)
    pr_infected = 1 - np.exp(-contact_rate / 1000)
    pr_noone_infected = binom.pmf(0, 1000, pr_infected)
    pr_extinction = pr_recovery * pr_noone_infected
    expected_extinctions = _find_num_successes(pr_extinction, TRIALS, ERROR_RATE)
    count_extinctions = 0
    for _ in range(TRIALS):
        model = CompartmentalModel(
            times=[0, 1],
            compartments=["S", "I", "R"],
            infectious_compartments=["I"],
        )
        model.set_initial_population(distribution={"S": 999, "I": 1})
        model.add_fractional_flow("recovery", recovery_rate, "I", "R")
        model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
        model.run_stochastic()
        is_extinct = model.outputs[1, 1] == 0
        if is_extinct:
            count_extinctions += 1

    assert count_extinctions >= expected_extinctions


DEATH_OR_RECOVERY_EXTINCTION_PARAMS = [
    # infect death rate, recovery rate, contact rate
    (0.2, 0.3, 2),
    (0.1, 0.1, 2),
    (0.01, 0.09, 1),
]


@pytest.mark.parametrize(
    "death_rate, recovery_rate, contact_rate", DEATH_OR_RECOVERY_EXTINCTION_PARAMS
)
def test_stochastic_exitinction(death_rate, recovery_rate, contact_rate):
    """
    A smokey test to make sure the disease goes extinct sometimes,
    because the infectious person recovers before they can infect someone else.

    Calculations similar to test_stochastic_death_exitinction
    """
    pr_death_or_recovery = 1 - np.exp(-1 * (death_rate + recovery_rate))
    pr_infected = 1 - np.exp(-contact_rate / 1000)
    pr_noone_infected = binom.pmf(0, 1000, pr_infected)
    pr_extinction = pr_death_or_recovery * pr_noone_infected
    expected_extinctions = _find_num_successes(pr_extinction, TRIALS, ERROR_RATE)
    count_extinctions = 0
    for _ in range(TRIALS):
        model = CompartmentalModel(
            times=[0, 1],
            compartments=["S", "I", "R"],
            infectious_compartments=["I"],
        )
        model.set_initial_population(distribution={"S": 999, "I": 1})
        model.add_death_flow("infect_death", death_rate, "I")
        model.add_fractional_flow("recovery", recovery_rate, "I", "R")
        model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
        model.run_stochastic()
        is_extinct = model.outputs[1, 1] == 0
        if is_extinct:
            count_extinctions += 1

    assert count_extinctions >= expected_extinctions


def _find_num_successes(p, N, e):
    """
    Returns the max number of successes guaranteed to happen with probability 1 - e (error)
    when we do N trials of a binomial with probability p of success.
    """

    def calc_pr_gte_k(k):
        """
        Returns probability to observe >= k successes in N trials
        """
        # Probability to observe <= k successes in N trials
        pr_lte_k = binom.cdf(k, N, p)
        # Probability to observe exactly k successes in N trials
        pr_k = binom.pmf(k, N, p)
        pr_lt_k = pr_lte_k - pr_k
        # Return probability we see k or more trials
        return 1 - pr_lt_k

    prev = 0
    for k in range(N):
        pr_gte_k = calc_pr_gte_k(k)
        if pr_gte_k + e < 1:
            return prev

        prev = k