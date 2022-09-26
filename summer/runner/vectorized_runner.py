from typing import Tuple

import numpy as np
import numba

import summer.flows as flows
from summer.compute import (
    accumulate_positive_flow_contributions,
    accumulate_negative_flow_contributions,
    sparse_pairs_accum,
    find_sum,
)

from summer.utils import get_model_param_value

from .model_runner import ModelRunner


class VectorizedRunner(ModelRunner):
    """
    An optimized, but less accessible model runner.
    """

    def __init__(self, model):
        super().__init__(model)

    def prepare_structural(self):
        super().prepare_structural()

        self.infectious_flow_indices = np.array(
            [i for i, f in self._iter_non_function_flows if isinstance(f, flows.BaseInfectionFlow)],
            dtype=int,
        )
        self.death_flow_indices = np.array(
            [i for i, f in self._iter_non_function_flows if f.is_death_flow], dtype=int
        )

        # Include dummy values in population_idx to account for Entry flows
        non_func_pops = np.array(
            [f.source.idx if f.source else 0 for i, f in self._iter_non_function_flows], dtype=int
        )

        func_pops = np.array(
            [f.source.idx if f.source else 0 for i, f in self._iter_function_flows], dtype=int
        )

        self.population_idx = np.concatenate((non_func_pops, func_pops))

        # Store indices of flows that are not population dependent
        self._non_pop_flow_idx = np.array(
            [
                i
                for i, f in self._iter_non_function_flows
                if (type(f) in (flows.ReplacementBirthFlow, flows.ImportFlow))
            ],
            dtype=int,
        )
        self._has_non_pop_flows = bool(len(self._non_pop_flow_idx))

        # Crude birth flows use population sum rather than a compartment; store indices here
        self._crude_birth_idx = np.array(
            [i for i, f in self._iter_non_function_flows if type(f) == flows.CrudeBirthFlow],
            dtype=int,
        )
        self._has_crude_birth = bool(len(self._crude_birth_idx))

        self._has_replacement = False
        # Replacement flows must be calculated after death flows, store indices here
        for i, f in self._iter_non_function_flows:
            if type(f) == flows.ReplacementBirthFlow:
                self._has_replacement = True
                self._replacement_flow_idx = i

    def prepare_dynamic(self, parameters: dict = None):
        """Do all precomputation here"""

        super().prepare_dynamic(parameters)

        self._precompute_flow_weights()
        self._precompute_flow_maps()

        self._build_infectious_multipliers_lookup()

        # Initialize arrays for infectious multipliers
        # FIXME Put this in its own function, this is getting messy again
        num_cats = self.num_categories
        num_strains = len(self.model._disease_strains)
        self._infection_density = {}
        self._infection_frequency = {}
        self._infection_density_flat = np.empty((num_strains, num_cats), dtype=float)
        self._infection_frequency_flat = np.empty((num_strains, num_cats), dtype=float)

    def _precompute_flow_weights(self):
        """Calculate all static flow weights before running,
        and build indices for time-varying weights"""
        self.flow_weights = np.zeros(len(self.model._flows))
        time_varying_weight_indices = []
        for i, f in self._iter_non_function_flows:
            weight_type = f.weight_type()
            if weight_type == flows.WeightType.STATIC:
                weight = f.get_weight_value(0, None, self.parameters)
                self.flow_weights[i] = weight
            elif weight_type == flows.WeightType.FUNCTION:
                time_varying_weight_indices.append(i)

        self.time_varying_weight_indices = np.array(time_varying_weight_indices, dtype=int)

        self._map_blocks()

    def _map_blocks(self):
        flow_block_maps = {}
        for i in self.time_varying_weight_indices:
            f = self.model._flows[i]
            key = (f.param, tuple(f.adjustments))
            if key not in flow_block_maps:
                flow_block_maps[key] = []
            flow_block_maps[key].append(i)

        self.flow_block_maps = dict(
            [(k, np.array(v, dtype=int)) for k, v in flow_block_maps.items()]
        )

    def _precompute_flow_maps(self):
        """Build fast-access arrays of flow indices"""
        f_pos_map = []
        f_neg_map = []
        for i, f in self._iter_non_function_flows:
            if f.source:
                f_neg_map.append((i, f.source.idx))
            if f.dest:
                f_pos_map.append((i, f.dest.idx))
        for i, f in self._iter_function_flows:
            if f.source:
                f_neg_map.append((i, f.source.idx))
            if f.dest:
                f_pos_map.append((i, f.dest.idx))

        self._pos_flow_map = np.array(f_pos_map, dtype=np.int)
        self._neg_flow_map = np.array(f_neg_map, dtype=np.int)

    def _prepare_time_step(self, time: float, compartment_values: np.ndarray):
        """
        Pre-timestep setup. This should be run before `_get_rates`.
        Here we set up any stateful updates that need to happen before we get the flow rates.
        """

        # Some flows (e.g birth replacement) expect this value to be defined
        self._timestep_deaths = 0.0

        # Find the effective infectious population for the force of infection (FoI) calculations.
        mixing_matrix = self._get_mixing_matrix(time)

        self._calculate_strain_infection_values(compartment_values, mixing_matrix)

    def _apply_flow_weights_at_time(self, time, computed_values):
        """Calculate time dependent flow weights and insert them into our weights array

        Args:
            time (float): Time in model.times coordinates
        """

        # Compute flow value blocks; these are 'chunked' blocks of flows that have a common
        # flow param/adjustment structure,
        # but are still specified as operating on individual flows
        for (param, adjustments), flow_idx in self.flow_block_maps.items():
            value = get_model_param_value(param, time, computed_values, self.parameters, True)
            for a in adjustments:
                value = a.get_new_value(value, computed_values, time, self.parameters)
            self.flow_weights[flow_idx] = value

    def _get_infectious_multipliers(self) -> np.ndarray:
        """Get multipliers for all infectious flows

        Returns:
            np.ndarray: Array of infectiousness multipliers
        """
        # We can use a fast lookup version if we have only one type of infectious multiplier
        # (eg only frequency, not mixed freq and density)
        if self._infection_frequency_only:
            return self._infection_frequency_flat[
                self._infect_strain_lookup_idx, self._infect_cat_lookup_idx
            ]
        elif self._infection_density_only:
            return self._infection_density_flat[
                self._infect_strain_lookup_idx, self._infect_cat_lookup_idx
            ]

        multipliers = np.empty(len(self.infectious_flow_indices))
        for i, idx in enumerate(self.infectious_flow_indices):
            f = self.model._flows[idx]
            multipliers[i] = f.find_infectious_multiplier(f.source, f.dest)
        return multipliers

    def _build_infectious_multipliers_lookup(self):
        """Get multipliers for all infectious flows

        These are used by _get_infectious_multipliers_flat (currently experimental)

        Returns:
            np.ndarray: Array of infectiousness multipliers
        """
        lookups = []

        has_freq = False
        has_dens = False

        for i, idx in enumerate(self.infectious_flow_indices):
            f = self.model._flows[idx]
            if isinstance(f, flows.InfectionFrequencyFlow):
                has_freq = True
            elif isinstance(f, flows.InfectionDensityFlow):
                has_dens = True
            cat_idx, strain = self._get_infection_multiplier_indices(f.source, f.dest)
            strain_idx = self.model._disease_strains.index(strain)
            lookups.append([strain_idx, cat_idx])
        full_table = np.array(lookups, dtype=int)
        self._full_table = full_table.reshape(len(self.infectious_flow_indices), 2)
        self._infect_strain_lookup_idx = self._full_table[:, 0].flatten()
        self._infect_cat_lookup_idx = self._full_table[:, 1].flatten()

        self._infection_frequency_only = False
        self._infection_density_only = False

        if has_freq and not has_dens:
            self._infection_frequency_only = True
        elif has_dens and not has_freq:
            self._infection_density_only = True

    def _calculate_strain_infection_values(
        self, compartment_values: np.ndarray, mixing_matrix: np.ndarray
    ):
        num_cats = self.num_categories
        # Calculate total number of people per category (for FoI).
        # A vector with size (num_cats).
        self._category_populations = sparse_pairs_accum(
            self._compartment_category_map, compartment_values, num_cats
        )

        for strain_idx, strain in enumerate(self.model._disease_strains):
            strain_compartment_infectiousness = self._compartment_infectiousness[strain]

            infection_density, infection_frequency = get_strain_infection_values(
                compartment_values,
                strain_compartment_infectiousness,
                self._compartment_category_map,
                num_cats,
                mixing_matrix,
                self._category_populations,
            )

            self._infection_density[strain] = infection_density
            self._infection_density_flat[strain_idx] = infection_density

            self._infection_frequency[strain] = infection_frequency
            self._infection_frequency_flat[strain_idx] = infection_frequency

    def _get_flow_rates(self, compartment_vals: np.ndarray, time: float) -> np.ndarray:
        """Get current flow rates, equivalent to calling get_net_flow on all (non-function) flows

        Args:
            comp_vals (np.ndarray): Compartment values
            time (float): Time in model.times coordinates

        Returns:
            np.ndarray: Array of all (non-function) flow rates
        """

        computed_values = self._calc_computed_values(compartment_vals, time)

        self._apply_flow_weights_at_time(time, computed_values)

        # These will be filled in afterwards
        populations = compartment_vals[self.population_idx]
        # Update for special cases (population-independent and CrudeBirth)
        if self._has_non_pop_flows:
            populations[self._non_pop_flow_idx] = 1.0
        if self._has_crude_birth:
            populations[self._crude_birth_idx] = find_sum(compartment_vals)

        flow_rates = self.flow_weights * populations

        # Calculate infection flows
        infect_mul = self._get_infectious_multipliers()
        flow_rates[self.infectious_flow_indices] *= infect_mul

        # ReplacementBirthFlow depends on death flows already being calculated; update here
        if self._has_replacement:
            # Only calculate timestep_deaths if we use replacement, it's expensive...
            self._timestep_deaths = flow_rates[self.death_flow_indices].sum()
            flow_rates[self._replacement_flow_idx] = self._timestep_deaths

        self._apply_function_flow_rates(compartment_vals, flow_rates, computed_values, time)

        return flow_rates

    def _apply_function_flow_rates(self, comp_vals, flow_rates, computed_values, time):
        if self._iter_function_flows:
            # Evaluate the function flows.
            for flow_idx, flow in self._iter_function_flows:
                net_flow = flow.get_net_flow(
                    self.model.compartments,
                    comp_vals,
                    self.model._flows,
                    flow_rates,
                    computed_values,
                    time,
                )
                flow_rates[flow_idx] = net_flow

    def _get_rates(self, comp_vals: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates inter-compartmental flow rates for a given state and time, as well
        as the updated compartment values once these rate deltas have been applied

        Args:
            comp_vals (np.ndarray): The current state of the model compartments
                                    (ie. number of people)
            time (float): Time in model.times coordinates

        Returns:
            Tuple[np.ndarray, np.ndarray]: (comp_rates, flow_rates) where
                comp_rates is the rate of change of compartments, and
                flow_rates is the contribution of each flow to compartment rate of change
        """
        self._prepare_time_step(time, comp_vals)

        comp_rates = np.zeros(len(comp_vals))
        flow_rates = self._get_flow_rates(comp_vals, time)

        if self._pos_flow_map.size > 0:
            accumulate_positive_flow_contributions(flow_rates, comp_rates, self._pos_flow_map)

        if self._neg_flow_map.size > 0:
            accumulate_negative_flow_contributions(flow_rates, comp_rates, self._neg_flow_map)

        return comp_rates, flow_rates

    def get_compartment_rates(self, compartment_values: np.ndarray, time: float):
        """
        Interface for the ODE solver: this function is passed to solve_ode func and defines the
        dynamics of the model.
        Returns the rate of change of the compartment values for a given state and time.
        """
        comp_vals = self._clean_compartment_values(compartment_values)
        comp_rates, _ = self._get_rates(comp_vals, time)
        return comp_rates

    def get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Returns the contribution of each flow to compartment rate of change for a given state and
        time.
        """
        comp_vals = self._clean_compartment_values(compartment_values)
        _, flow_rates = self._get_rates(comp_vals, time)
        return flow_rates


"""
Additional functions - fast JIT specializations etc
"""


@numba.jit
def get_strain_infection_values(
    compartment_values,
    strain_compartment_infectiousness,
    compartment_category_map,
    num_cats,
    mixing_matrix,
    category_populations,
):
    infected_values = compartment_values * strain_compartment_infectiousness
    infectious_populations = sparse_pairs_accum(compartment_category_map, infected_values, num_cats)
    infection_density = mixing_matrix @ infectious_populations
    category_prevalence = infectious_populations / category_populations
    infection_frequency = mixing_matrix @ category_prevalence

    return infection_density, infection_frequency
