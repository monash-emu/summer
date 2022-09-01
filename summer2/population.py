"""Utilities for population redistribution
"""

from typing import NamedTuple

import numpy as np

from summer2.parameters import get_static_param_value


class CompartmentGroup(NamedTuple):
    """Hashable compartment groupings"""

    name: str
    strata: frozenset


def get_unique_strat_groups(comps, strat):
    """
    Return a set of unique name,strata for the given set of compartments,
    which differ only by the strata for 'strat', ie the set of sets
    whose children are specified by 'strat'
    """
    unique_strat_groups = set()
    for c in comps:
        cur_strata = c.strata.copy()
        cur_strata.pop(strat)
        unique_strat_groups.add(CompartmentGroup(c.name, frozenset(cur_strata.items())))
    return unique_strat_groups


def filter_by_strata(comps, strata):
    _strata = frozenset(strata.items())
    return [c for c in comps if c._has_strata(_strata)]


def get_rebalanced_population(
    model,
    population: np.ndarray,
    static_graph_values: dict,
    strat: str,
    dest_filter: dict,
    proportions: dict,
):
    """Adjust the initial population to redistribute the population for a particular
    stratification, over a subset of some other strata

    Args:
        model: The CompartmentalModel to adjust
        strat (str): The stratification to redistribute over
        dest_filter (dict): Subset of (other) strata to filter the split by
        proportions (dict): Proportions of new split (must have all strata specified)

    """

    msg = f"No stratification {strat} found in model"
    assert strat in [s.name for s in model._stratifications], msg

    model_strat = [s for s in model._stratifications if s.name == strat][0]

    proportions = get_static_param_value(proportions, static_graph_values)

    msg = "All strata must be specified in proportions"
    assert set(model_strat.strata) == set(proportions), msg

    msg = "Proportions must sum to 1.0"
    np.testing.assert_allclose(sum(proportions.values()), 1.0, err_msg=msg)

    strat_comps = [c for c in model.compartments if strat in c.strata]
    # Filter by only the subset we're setting in split_map
    strat_comps = filter_by_strata(strat_comps, dest_filter)

    usg = get_unique_strat_groups(strat_comps, strat)

    out_population = population.copy()

    for g in usg:
        mcomps = model._get_matching_compartments(g.name, g.strata)
        idx = np.array([c.idx for c in mcomps])
        total = population[idx].sum()
        for c in mcomps:
            k = c.strata[strat]
            target_prop = proportions[k]
            out_population = out_population.at[c.idx].set(total * target_prop)

    return out_population


def calculate_initial_population(model, parameters=None) -> np.ndarray:
    """
    Called to recalculate the initial population from either fixed dictionary, or a dict
    supplied as a parameter
    """
    if parameters is None:
        parameters = {}
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
            pop = get_static_param_value(distribution[comp.name], parameters, True)
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
