import numpy as np

import pytest

from summer import CompartmentalModel
from summer.stratification import Stratification

def build_model():
    """Returns a model for the stratification examples"""
    model = CompartmentalModel(
        times=[0, 20],
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
        timestep=0.1,
    )

    # Add people to the model
    model.set_initial_population(distribution={"S": 990, "I": 10})

    strata = ["young", "old"]
    age_strat = Stratification(name="age", strata=strata, compartments=["S", "I", "R"])
    age_strat.set_population_split({"young": 0.6, "old": 0.4})

    strata = ["urban", "rural"]
    loc_strat = Stratification(name="loc", strata=strata, compartments=["S","I","R"])
    loc_strat.set_population_split({"urban": 0.8, "rural": 0.2})

    strata = ["one_dose", "two_dose", "unvacc"]
    vacc_strat = Stratification(name="vacc", strata=strata, compartments=["S","I", "R"])

    model.stratify_with(age_strat)
    model.stratify_with(loc_strat)
    model.stratify_with(vacc_strat)
    return model

def test_split_single_filter():
    model = build_model()

    s_young_orig = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"age": "young"})]].copy()

    model.adjust_population_split("vacc", {"age": "old"}, {'one_dose': 0.7, "two_dose": 0.2, "unvacc": 0.1})
    
    # Check the total population hasn't changed
    np.testing.assert_almost_equal(model.initial_population.sum(), 1000.0)

    # Check we haven't changed any compartments we're not meant to

    s_young_post = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"age": "young"})]]
    np.testing.assert_array_almost_equal(s_young_orig, s_young_post)
    
    # Check we got the expected rebalance values
    s_old_urban_expected = 990.0 * 0.4 * 0.8 * np.array((0.7, 0.2, 0.1))
    s_old_urban_target = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"age": "old", "loc": "urban"})]]

    np.testing.assert_array_almost_equal(s_old_urban_expected, s_old_urban_target)

def test_split_multi_filter():
    model = build_model()

    s_young_orig = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"age": "young"})]].copy()
    s_rural_orig = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"loc": "rural"})]].copy()

    model.adjust_population_split("vacc", {"age": "old", "loc": "urban"}, {'one_dose': 0.7, "two_dose": 0.2, "unvacc": 0.1})
    
    # Check the total population hasn't changed
    np.testing.assert_almost_equal(model.initial_population.sum(), 1000.0)

    # Check we haven't changed any compartments we're not meant to

    s_young_post = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"age": "young"})]]
    s_rural_post = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"loc": "rural"})]]
    np.testing.assert_array_almost_equal(s_young_orig, s_young_post)
    np.testing.assert_array_almost_equal(s_rural_orig, s_rural_post)
    
    # Check we got the expected rebalance values
    s_old_urban_expected = 990.0 * 0.4 * 0.8 * np.array((0.7, 0.2, 0.1))
    s_old_urban_target = model.initial_population[[c.idx for c in model.get_matching_compartments("S", {"age": "old", "loc": "urban"})]]

    np.testing.assert_array_almost_equal(s_old_urban_expected, s_old_urban_target)

def test_expected_failures():
    model = build_model()

    with pytest.raises(AssertionError, match="No stratification shoes found in model"):
        model.adjust_population_split("shoes", {"age": "old", "loc": "urban"}, {'one_dose': 0.7, "two_dose": 0.2, "unvacc": 0.1})
    
    with pytest.raises(AssertionError, match="All strata must be specified in proportions"):
        model.adjust_population_split("vacc", {"age": "old", "loc": "urban"}, {'one_dose': 0.7, "two_dose": 0.2})
    
    with pytest.raises(AssertionError, match="Proportions must sum to 1.0"):
        model.adjust_population_split("vacc", {"age": "old", "loc": "urban"}, {'one_dose': 0.7, "two_dose": 0.5, "unvacc": 0.8})
    