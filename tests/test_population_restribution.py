import numpy as np

import pytest

from summer2 import CompartmentalModel
from summer2.stratification import Stratification

from summer2.population import calculate_initial_population


def build_model():
    """Returns a model for the stratification examples"""
    model = CompartmentalModel(
        times=[0, 1],
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
    )

    # Add people to the model
    model.set_initial_population(distribution={"S": 990, "I": 10})

    strata = ["young", "old"]
    age_strat = Stratification(name="age", strata=strata, compartments=["S", "I", "R"])
    age_strat.set_population_split({"young": 0.6, "old": 0.4})

    strata = ["urban", "rural"]
    loc_strat = Stratification(name="loc", strata=strata, compartments=["S", "I", "R"])
    loc_strat.set_population_split({"urban": 0.8, "rural": 0.2})

    strata = ["one_dose", "two_dose", "unvacc"]
    vacc_strat = Stratification(name="vacc", strata=strata, compartments=["S", "I", "R"])

    model.stratify_with(age_strat)
    model.stratify_with(loc_strat)
    model.stratify_with(vacc_strat)

    return model


def test_split_single_filter():
    model = build_model()

    orig_init_pop = model._get_step_test()["initial_population"]
    s_young_orig = orig_init_pop[
        model.query_compartments({"name": "S", "age": "young"},as_idx=True)
    ].copy()

    

    model = build_model()
    model.adjust_population_split(
        "vacc", {"age": "old"}, {"one_dose": 0.7, "two_dose": 0.2, "unvacc": 0.1}
    )

    init_pop = model._get_step_test()["initial_population"]

    # Check the total population hasn't changed
    np.testing.assert_almost_equal(init_pop.sum(), 1000.0)

    # Check we haven't changed any compartments we're not meant to

    s_young_post = init_pop[model.query_compartments({"name": "S", "age": "young"},as_idx=True)]
    np.testing.assert_array_almost_equal(s_young_orig, s_young_post)

    # Check we got the expected rebalance values
    s_old_urban_expected = 990.0 * 0.4 * 0.8 * np.array((0.7, 0.2, 0.1))
    s_old_urban_target = init_pop[
        model.query_compartments({"name": "S", "age": "old", "loc": "urban"},as_idx=True)
    ]

    np.testing.assert_array_almost_equal(s_old_urban_expected, s_old_urban_target)


def test_split_multi_filter():
    model = build_model()

    init_pop_orig = model._get_step_test()["initial_population"]

    s_young_orig = init_pop_orig[
        model.query_compartments({"name": "S", "age": "young"},as_idx=True)
    ].copy()
    s_rural_orig = init_pop_orig[
        model.query_compartments({"name": "S", "loc": "rural"},as_idx=True)
    ].copy()

    model = build_model()
    model.adjust_population_split(
        "vacc", {"age": "old", "loc": "urban"}, {"one_dose": 0.7, "two_dose": 0.2, "unvacc": 0.1}
    )

    init_pop = model._get_step_test()["initial_population"]
    # Check the total population hasn't changed
    np.testing.assert_almost_equal(init_pop.sum(), 1000.0)

    # Check we haven't changed any compartments we're not meant to

    s_young_post = init_pop[model.query_compartments({"name": "S", "age": "young"},as_idx=True)]
    s_rural_post = init_pop[model.query_compartments({"name": "S", "loc": "rural"},as_idx=True)]
    np.testing.assert_array_almost_equal(s_young_orig, s_young_post)
    np.testing.assert_array_almost_equal(s_rural_orig, s_rural_post)

    # Check we got the expected rebalance values
    s_old_urban_expected = 990.0 * 0.4 * 0.8 * np.array((0.7, 0.2, 0.1))
    s_old_urban_target = init_pop[
        model.query_compartments({"name": "S", "age": "old", "loc": "urban"},as_idx=True)
    ]

    np.testing.assert_array_almost_equal(s_old_urban_expected, s_old_urban_target)


def test_expected_failures():
    model = build_model()

    with pytest.raises(AssertionError, match="No stratification shoes found in model"):
        model.adjust_population_split(
            "shoes",
            {"age": "old", "loc": "urban"},
            {"one_dose": 0.7, "two_dose": 0.2, "unvacc": 0.1},
        )

    with pytest.raises(AssertionError, match="All strata must be specified in proportions"):
        model.adjust_population_split(
            "vacc", {"age": "old", "loc": "urban"}, {"one_dose": 0.7, "two_dose": 0.2}
        )

    with pytest.raises(AssertionError, match="Proportions must sum to 1.0"):
        model.adjust_population_split(
            "vacc",
            {"age": "old", "loc": "urban"},
            {"one_dose": 0.7, "two_dose": 0.5, "unvacc": 0.8},
        )
