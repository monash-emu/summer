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


def test_set_times():

    # Acceptable and non-acceptable requests, even though some may not be that sensible.
    good_time_requests = [
        [0, 5, 1],
        [1, 8, 1],
        [2, 9, 2],
        [-10, 5, 4],
        [0, 10, 100],
    ]
    bad_time_requests = [
        [5, 0, 1],
        [8, 1, 1],
        [0, -5, 2],
    ]

    for time_request in good_time_requests:
        start_time, end_time, time_step = time_request
        model = CompartmentalModel(
            times=(start_time, end_time),
            compartments=["S", "I", "R"],
            infectious_compartments=["I"],
            timestep=time_step
        )
        assert all([start_time <= t <= end_time for t in model.times])
        assert all(np.sort(model.times) == model.times), "Model times not sorted"
        assert start_time in model.times  # Start time should always be included in evaluation times.
        assert end_time in model.times  # End Time should always be included in evaluation times.
        # Check time step has been applied correctly
        for i_time in range(2, len(model.times) - 1):
            assert model.times[i_time] - model.times[i_time - 1] == float(time_request[2])

    for time_request in bad_time_requests:
        start_time, end_time, time_step = time_request
        with pytest.raises(AssertionError):
            model = CompartmentalModel(
                times=(start_time, end_time),
                compartments=["S", "I", "R"],
                infectious_compartments=["I"],
                timestep=time_step
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
