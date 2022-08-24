"""
General utility functions used accross submodules
"""

import pandas as pd
from datetime import timedelta, datetime
from typing import Iterable
import numpy as np


def get_scenario_start_index(base_times, start_time):
    """
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    assert (
        base_times[0] <= start_time
    ), f"Scenario start time {start_time} is before baseline has started"
    indices_after_start_index = [idx for idx, time in enumerate(base_times) if time > start_time]
    if not indices_after_start_index:
        raise ValueError(f"Scenario start time {start_time} is set after the baseline time range")

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index


def ref_times_to_dti(ref_date: datetime, times: Iterable) -> pd.DatetimeIndex:
    """Return a DatetimeIndex whose values after ref_date offset by times

    Args:
        ref_date (datetime): The reference  date ('epoch') from which offsets are computed
        times (Iterable): Offsets to ref_date (int or float)

    Returns:
        (pd.DatetimeIndex): Index suitable for constructing Series or DataFrame
    """

    return pd.DatetimeIndex([ref_date + timedelta(t) for t in times])


def calculate_initial_population(model, parameters=None) -> np.ndarray:
    """
    Called to recalculate the initial population from either fixed dictionary, or a dict
    supplied as a parameter
    """
    # FIXME:
    # Work in progress; correctly recalculates non-parameterized
    # populations, but does not include population rebalances etc
    distribution = model._init_pop_dist
    initial_population = np.zeros_like(model._original_compartment_names, dtype=float)

    # if is_var(distribution, "parameters"):
    #    distribution = self.parameters[distribution.name]
    # elif isinstance(distribution, Function) or isinstance(distribution, ModelParameter):
    #    distribution = get_static_param_value(distribution, parameters)

    if isinstance(distribution, dict):
        for idx, comp in enumerate(model._original_compartment_names):
            pop = get_static_param_value(distribution[comp.name], parameters)
            assert pop >= 0, f"Population for {comp.name} cannot be negative: {pop}"
            initial_population[idx] = pop

        comps = model._original_compartment_names

        for action in model.tracker.all_actions:
            if action.action_type == "stratify":
                strat = action.kwargs["strat"]
                # for strat in self.model._stratifications:
                # Stratify compartments, split according to split_proportions
                prev_compartment_names = comps  # copy.copy(self.compartments)
                comps = strat._stratify_compartments(comps)
                initial_population = strat._stratify_compartment_values(
                    prev_compartment_names, initial_population, parameters
                )
            elif action.action_type == "adjust_pop_split":
                initial_population = get_rebalanced_population(
                    model, initial_population, parameters, **action.kwargs
                )
        return initial_population
    else:
        raise TypeError(
            "Initial population distribution must be a dict",
            distribution,
        )
