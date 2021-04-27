import numpy as np
import pytest
from numpy.testing import assert_array_equal

from summer import Compartment, CompartmentalModel


def test_create_model():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert_array_equal(model.times, np.array([0, 1, 2, 3, 4, 5]))
    assert model.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    assert model._infectious_compartments == [Compartment("I")]
    assert_array_equal(model.initial_population, np.array([0, 0, 0]))

    # Times out of order
    with pytest.raises(AssertionError):
        CompartmentalModel(
            times=[5, 0], compartments=["S", "I", "R"], infectious_compartments=["I"]
        )

    # Infectious compartment not a compartment
    with pytest.raises(AssertionError):
        CompartmentalModel(
            times=[-1, 5], compartments=["S", "I", "R"], infectious_compartments=["E"]
        )


SUCCESS_TIME_REQUESTS = [
    # start_time, end_time, time_step
    [0, 5, 1],  # Regular request
    [1, 8, 1],  # Nonzero start time
    [2, 9, 2],  # Timestep > 1
    [2, 9, 1.2345678],  # Timestep float > 1
    [2, 9, 3.14159265358979323846],  # Timestep < 1
    [-10, 5, 4],  # Negative start time
    [0, 10, 100],  # Very large timestep
    [0.5, 3.5, 0.1],  # Non integer start and end
]


@pytest.mark.parametrize("start_time, end_time, time_step", SUCCESS_TIME_REQUESTS)
def test_set_times__with_success(start_time, end_time, time_step):
    model = CompartmentalModel(
        times=(start_time, end_time),
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
        timestep=time_step,
    )
    assert all([start_time <= t <= end_time for t in model.times])
    assert_array_equal(model.times, np.sort(model.times))
    assert_array_equal(model.times, np.unique(model.times))
    assert start_time in model.times  # Start time should always be included in evaluation times.
    assert end_time in model.times  # End Time should always be included in evaluation times.
    # Check time step size has been applied correctly for each time step
    for i_time in range(2, len(model.times) - 1):
        assert abs(model.times[i_time] - model.times[i_time - 1] - time_step) < 1e-6


FAIL_TIME_REQUESTS = [
    # start_time, end_time, time_step
    [5, 0, 1],  # End time after start time
    [0, -5, 1],  # End time after start time
]


@pytest.mark.parametrize("start_time, end_time, time_step", FAIL_TIME_REQUESTS)
def test_set_times__with_failure(start_time, end_time, time_step):
    with pytest.raises(AssertionError):
        CompartmentalModel(
            times=(start_time, end_time),
            compartments=["S", "I", "R"],
            infectious_compartments=["I"],
            timestep=time_step,
        )


def test_set_initial_population():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert_array_equal(model.initial_population, np.array([0, 0, 0]))
    model.set_initial_population({"S": 100})
    assert_array_equal(model.initial_population, np.array([100, 0, 0]))
    model.set_initial_population({"I": 100})
    assert_array_equal(model.initial_population, np.array([0, 100, 0]))
    model.set_initial_population({"R": 1, "S": 50, "I": 99})
    assert_array_equal(model.initial_population, np.array([50, 99, 1]))
