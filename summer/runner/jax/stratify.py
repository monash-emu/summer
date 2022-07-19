# Reimplementation of stratification and initial pop calculation for Jax runners

from __future__ import annotations
from typing import List, TYPE_CHECKING

from jax import numpy as jnp

from summer.parameters import get_static_param_value, is_var
from summer.stratification import Stratification

if TYPE_CHECKING:
    from summer import CompartmentalModel


def get_stratify_compartments_func(strat: Stratification, input_comps: List[str]):
    """Build the equivalent of self._stratify_compartment_values for a given Stratification

    Args:
        strat (Stratification): The stratification to build this for
        input_comps (List[str]): The list of input compartments (pre-stratification)
    """

    def stratify_compartment_values(comp_values: jnp.ndarray, parameters: dict = None):
        """
        Stratify the model compartments into sub-compartments, based on the strata names provided.
        Split the population according to the provided proportions.
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartment values.
        """
        new_comp_values = []

        population_split = get_static_param_value(strat.population_split, parameters)

        for idx in range(len(comp_values)):
            should_stratify = input_comps[idx].has_name_in_list(strat.compartments)
            if should_stratify:
                for stratum in strat.strata:
                    new_value = comp_values[idx] * population_split[stratum]
                    new_comp_values.append(new_value)
            else:
                new_comp_values.append(comp_values[idx])

        return jnp.array(new_comp_values)

    return stratify_compartment_values


def get_calculate_initial_pop(model: CompartmentalModel):
    """This is the function we're most interested in

    Args:
        model (_type_): _description_

    Raises:
        NotImplementedError: _description_
        TypeError: _description_

    Returns:
        _type_: _description_
    """

    strat_funcs = {}
    comps = model._original_compartment_names
    for strat in model._stratifications:
        strat_funcs[strat] = get_stratify_compartments_func(strat, comps)
        comps = strat._stratify_compartments(comps)

    def calculate_initial_population(parameters: dict) -> jnp.ndarray:
        """
        Called to recalculate the initial population from either fixed dictionary, or a dict
        supplied as a parameter
        """
        # FIXME:
        # Work in progress; correctly recalculates non-parameterized
        # populations, but does not include population rebalances etc
        distribution = model._initial_population_distribution
        initial_population = jnp.zeros(len(model._original_compartment_names))

        if is_var(distribution, "parameters"):
            distribution = parameters[distribution.name]

        if isinstance(distribution, dict):
            for idx, comp in enumerate(model._original_compartment_names):
                pop = distribution.get(comp.name, 0)
                initial_population = initial_population.at[idx].set(pop)

            for action in model.tracker.all_actions:
                if action.action_type == "stratify":
                    strat = action.kwargs["strat"]
                    initial_population = strat_funcs[strat](initial_population, parameters)
                elif action.action_type == "adjust_pop_split":
                    # FIXME: Implement this
                    raise NotImplementedError
                    # initial_population = get_rebalanced_population(
                    #    model, initial_population, parameters, **action.kwargs
                    # )
            return initial_population
        else:
            raise TypeError(
                "Initial population distribution must be a dict or a Function that returns one",
                distribution,
            )

    return calculate_initial_population
