# Reimplementation of stratification and initial pop calculation for Jax runners

from __future__ import annotations
from typing import List, TYPE_CHECKING

from jax import numpy as jnp

from summer2.parameters import get_static_param_value, is_var, Function
from summer2.parameters.param_impl import ModelParameter
from summer2.stratification import Stratification

from summer2.population import get_rebalanced_population

if TYPE_CHECKING:
    from summer2 import CompartmentalModel


def get_stratify_compartments_func(
    model: CompartmentalModel, strat: Stratification, input_comps: List[str]
):
    """Build the equivalent of self._stratify_compartment_values for a given Stratification

    Args:
        strat (Stratification): The stratification to build this for
        input_comps (List[str]): The list of input compartments (pre-stratification)
    """

    def _stratify_compartment_values(comp_values: jnp.ndarray, static_graph_values: dict = None):
        """
        Stratify the model compartments into sub-compartments, based on the strata names provided.
        Split the population according to the provided proportions.
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartment values.
        """
        new_comp_values = []

        population_split = get_static_param_value(strat.population_split, static_graph_values)

        for idx in range(len(comp_values)):
            should_stratify = input_comps[idx].has_name_in_list(strat.compartments)
            if should_stratify:
                for stratum in strat.strata:
                    new_value = comp_values[idx] * population_split[stratum]
                    new_comp_values.append(new_value)
            else:
                new_comp_values.append(comp_values[idx])

        return jnp.array(new_comp_values)

    def stratify_compartment_values(comp_values: jnp.ndarray, static_graph_values: dict = None):
        """
        Stratify the model compartments into sub-compartments, based on the strata names provided.
        Split the population according to the provided proportions.
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartment values.
        """
        population_split = get_static_param_value(strat.population_split, static_graph_values)

        new_comp_values = jnp.empty(strat._new_size)
        new_comp_values = new_comp_values.at[strat._passthrough_target_indices].set(
            comp_values[strat._passthrough_base_indices]
        )

        base_values = comp_values[strat._strat_base_indices]
        for stratum in strat.strata:
            new_value = base_values * population_split[stratum]
            new_comp_values = new_comp_values.at[strat._stratum_target_indices[stratum]].set(
                new_value
            )
        return new_comp_values

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
        strat_funcs[strat] = get_stratify_compartments_func(model, strat, comps)
        comps = strat._stratify_compartments(comps)

    def calculate_initial_population(static_graph_values: dict) -> jnp.ndarray:
        """
        Called to recalculate the initial population from either fixed dictionary, or a dict
        supplied as a parameter
        """
        # FIXME:
        # Work in progress; correctly recalculates non-parameterized
        # populations, but does not include population rebalances etc
        distribution = model._init_pop_dist
        initial_population = jnp.zeros(len(model._original_compartment_names))

        if isinstance(distribution, dict):
            for idx, comp in enumerate(model._original_compartment_names):
                pop = get_static_param_value(distribution[comp.name], static_graph_values)
                initial_population = initial_population.at[idx].set(pop)

            for action in model.tracker.all_actions:
                if action.action_type == "stratify":
                    strat = action.kwargs["strat"]
                    initial_population = strat_funcs[strat](initial_population, static_graph_values)
                elif action.action_type == "adjust_pop_split":
                    # FIXME: Implement this
                    initial_population = get_rebalanced_population(
                        model, initial_population, static_graph_values, **action.kwargs
                    )
            return initial_population
        else:
            raise TypeError(
                "Initial population distribution must be a dict or a Function that returns one",
                distribution,
            )

    return calculate_initial_population
