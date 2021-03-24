import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from summer import CompartmentalModel

RANDOM_SEED = 1337


TRANSITION_PARAMS = [
    # pop, rtol, recovery_rate
    (1e2, 0.8, 0.02),
    # if we increase population, we should expect tighter %error (law of large numbers)
    (1e4, 0.1, 0.02),
    (1e7, 0.011, 0.02),
]


@pytest.mark.parametrize("pop, rtol, recovery_rate", TRANSITION_PARAMS)
def test_stochastic_transition_flows(pop, rtol, recovery_rate):
    """
    Check that transition flows produce outputs that tend towards mean as pop increases.
    """
    model = CompartmentalModel(
        times=[0, 10], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    s_pop = 0.10 * pop
    i_pop = 0.90 * pop
    model.set_initial_population(distribution={"S": s_pop, "I": i_pop})
    model.add_fractional_flow("recovery", recovery_rate, "I", "R")
    model.run_stochastic(RANDOM_SEED)

    # No change to susceptible compartments
    assert_array_equal(model.outputs[:, 0], s_pop)

    # Calculate recoveries using mean recovery rate
    mean_i = np.zeros_like(model.times)
    mean_r = np.zeros_like(model.times)
    mean_i[0] = i_pop
    for i in range(1, len(model.times)):
        recovered = mean_i[i - 1] * recovery_rate
        mean_i[i] = mean_i[i - 1] - recovered
        mean_r[i] = mean_r[i - 1] + recovered

    # All I and R compartment sizes are are within the error range
    # of the mean of the multinomial dist that determines transition.
    assert_allclose(model.outputs[:, 1], mean_i, rtol=rtol)
    assert_allclose(model.outputs[:, 2], mean_r, rtol=rtol)


EXIT_PARAMS = [
    # pop, rtol, deathrate
    (1e3, 0.2, 0.02),
    # if we increase population, we should expect tighter %error (law of large numbers)
    (1e4, 0.04, 0.02),
    (1e7, 0.003, 0.02),
]


@pytest.mark.parametrize("pop, rtol, deathrate", EXIT_PARAMS)
def test_stochastic_exit_flows(pop, rtol, deathrate):
    """
    Check that death flows produce outputs that tend towards mean as pop increases.
    """
    model = CompartmentalModel(
        times=[0, 10], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    s_pop = 0.80 * pop
    i_pop = 0.20 * pop
    model.set_initial_population(distribution={"S": s_pop, "I": i_pop})
    model.add_universal_death_flows("deaths", deathrate)
    model.run_stochastic(RANDOM_SEED)

    # No change to recovered compartments
    assert_array_equal(model.outputs[:, 2], 0)

    # Calculate births using mean birth rate
    mean_s = np.zeros_like(model.times)
    mean_i = np.zeros_like(model.times)
    mean_s[0] = s_pop
    mean_i[0] = i_pop
    for i in range(1, len(model.times)):
        mean_s[i] = mean_s[i - 1] - deathrate * mean_s[i - 1]
        mean_i[i] = mean_i[i - 1] - deathrate * mean_i[i - 1]

    # All S and I compartment sizes are are within the error range
    # of the mean of the multinomial dist that determines exit.
    assert_allclose(model.outputs[:, 0], mean_s, rtol=rtol)
    assert_allclose(model.outputs[:, 1], mean_i, rtol=rtol)


ENTRY_PARAMS = [
    # pop, rtol, birthrate
    (1e2, 0.1, 0.02),
    # if we increase population, we should expect tighter %error (law of large numbers)
    (1e4, 0.02, 0.02),
    (1e7, 0.0004, 0.02),
]


@pytest.mark.parametrize("pop, rtol, birthrate", ENTRY_PARAMS)
def test_stochastic_entry_flows(pop, rtol, birthrate):
    """
    Check that entry flow produces outputs that tend towards mean as pop increases.
    """
    model = CompartmentalModel(
        times=[0, 10], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    s_pop = 0.99 * pop
    i_pop = 0.01 * pop
    model.set_initial_population(distribution={"S": s_pop, "I": i_pop})
    model.add_crude_birth_flow("births", birthrate, "S")
    model.run_stochastic(RANDOM_SEED)

    # No change to infected or recovered compartments
    assert_array_equal(model.outputs[:, 1], i_pop)
    assert_array_equal(model.outputs[:, 2], 0)

    # Calculate births using mean birth rate
    mean_s = np.zeros_like(model.times)
    mean_s[0] = s_pop
    for i in range(1, len(model.times)):
        mean_s[i] = mean_s[i - 1] + birthrate * (i_pop + mean_s[i - 1])

    # All S compartment sizes are are within the error range
    # of the mean of the poisson dist that determines entry.
    assert_allclose(model.outputs[:, 0], mean_s, rtol=rtol)
