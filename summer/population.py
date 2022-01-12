"""Utilities for population redistribution
"""

from typing import NamedTuple

class CompartmentGroup(NamedTuple):
    """Hashable compartment groupings
    """
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
        unique_strat_groups.add(CompartmentGroup(c.name,frozenset(cur_strata.items())))
    return unique_strat_groups

def filter_by_strata(comps, strata):
    _strata = frozenset(strata.items())
    return [c for c in comps if c._has_strata(_strata)]
