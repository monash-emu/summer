import pytest

from summer import CompartmentalModel, Stratification, AgeStratification


def test_full_age_strat():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    age_strat = AgeStratification(name="age", strata=["0", "5", "10"], compartments=["S", "I", "R"])
    model.stratify_with(age_strat)


def test_partial_age_strat_fails():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    age_strat = AgeStratification(name="age", strata=["0", "5", "10"], compartments=["I", "R"])

    with pytest.raises(AssertionError):
        model.stratify_with(age_strat)


def test_repeat_age_strat_fails():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    age_strat = AgeStratification(name="age", strata=["0", "5", "10"], compartments=["S", "I", "R"])
    model.stratify_with(age_strat)
    with pytest.raises(AssertionError):
        model.stratify_with(age_strat)


def test_repeat_strat_including_age():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    age_strat = AgeStratification(name="age", strata=["0", "5", "10"], compartments=["S", "I", "R"])
    other_strat = Stratification(name="gender", strata=["female", "male"], compartments=["S", "I", "R"])
    model.stratify_with(age_strat)
    model.stratify_with(other_strat)
