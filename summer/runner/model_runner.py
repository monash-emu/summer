from typing import Tuple

import numpy as np
import numba

from summer import Compartment
from summer.adjust import Overwrite
import summer.flows as flows
from summer.compute import (
    accumulate_positive_flow_contributions,
    accumulate_negative_flow_contributions,
    sparse_pairs_accum,
    find_sum,
    binary_matrix_to_sparse_pairs,
)
from summer.utils import clean_compartment_values


class ModelRunner:
    """
    An optimized, but less accessible model runner.
    """

    def __init__(self, model):
        # Compartmental model
        self.model = model

        # Tracks total deaths per timestep for death-replacement birth flows
        self._timestep_deaths = None

        # Set our initial parameters to an empty dict - this is really just to appease tests
        self.parameters = {}

    def prepare_structural(self):
        # FIXME: Redundant
        self._iter_non_function_flows = [(i, f) for i, f in enumerate(self.model._flows)]

        self._build_compartment_category_map()

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

        self.population_idx = non_func_pops

        # Store indices of flows that are not population dependent
        self._non_pop_flow_idx = np.array(
            [
                i
                for i, f in self._iter_non_function_flows
                if (type(f) in (flows.ReplacementBirthFlow, flows.ImportFlow, flows.AbsoluteFlow))
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

        self._precompute_flow_maps()
        self._build_infectious_multipliers_lookup()

    def prepare_dynamic(self, parameters: dict = None):
        """Do all precomputation here"""

        self.parameters = parameters

        #
        source_inputs = {"parameters": parameters}

        # Can replace this with calibration parameters later
        dyn_vars = self.model.graph.get_input_variables()

        # Graph frozen for all non-calibration parameters
        self.param_frozen_cg = self.model.graph.freeze(dyn_vars, source_inputs)

        # Query model_variables (ie time-varying sources)
        ts_vars = self.param_frozen_cg.query("model_variables")

        # Subgraph that does not contain model_variables (ie anything time varying)
        static_cg = self.param_frozen_cg.filter(exclude=ts_vars)
        self._graph_values_static = static_cg.get_callable()(parameters=parameters)

        # Subgraph whose only dynamic inputs are model_variables
        self.timestep_cg = self.param_frozen_cg.freeze(ts_vars, source_inputs)

        self.run_graph_ts = self.timestep_cg.get_callable()

        # Must be run after self._graph_values_static is available
        self.calculate_initial_population()

        self._compartment_infectiousness = {
            strain_name: self._get_compartment_infectiousness_for_strain(strain_name)
            for strain_name in self.model._disease_strains
        }

        # Initialize arrays for infectious multipliers
        # FIXME Put this in its own function, this is getting messy again
        num_cats = self.num_categories
        num_strains = len(self.model._disease_strains)
        self._infection_density = {}
        self._infection_frequency = {}
        self._infection_density_flat = np.empty((num_strains, num_cats), dtype=float)
        self._infection_frequency_flat = np.empty((num_strains, num_cats), dtype=float)

    def prepare_to_run(self, parameters: dict = None):
        # FIXME:
        # This is now only used by tests, and should never called in any production code
        if not hasattr(self.model, "_init_pop_dist"):
            self.model.set_initial_population({}, force=True)
        self.model.finalize()
        self.prepare_structural()
        self.prepare_dynamic(parameters)

    def _build_compartment_category_map(self):
        # Create a matrix that tracks which categories each compartment is in.
        # A matrix with size (num_cats x num_comps).
        # This is a very sparse static matrix, so we'll flatten it afterwards
        num_comps = len(self.model.compartments)
        self.num_categories = len(self.model._mixing_categories)
        self._category_lookup = {}  # Map compartments to categories.
        self._category_matrix = np.zeros((self.num_categories, num_comps))
        for i, category in enumerate(self.model._mixing_categories):
            for j, comp in enumerate(self.model.compartments):
                if all(comp.has_stratum(k, v) for k, v in category.items()):
                    self._category_matrix[i][j] = 1
                    self._category_lookup[j] = i

        # This is the actual structure we use to compute lookups later on
        # It's still not optimal, but it'll do...
        self._compartment_category_map = binary_matrix_to_sparse_pairs(self._category_matrix)

    def _get_compartment_infectiousness_for_strain(self, strain: str):
        """
        Returns a vector of floats, each representing the relative infectiousness of each
        compartment.
        If a strain name is provided, find the infectiousness factor *only for that strain*.
        """
        # Figure out which compartments should be infectious
        infectious_mask = np.array(
            [
                c.has_name_in_list(self.model._infectious_compartments)
                for c in self.model.compartments
            ]
        )
        # Find the infectiousness multipliers for each compartment being implemented in the model.
        # Start from assumption that each compartment is not infectious.
        compartment_infectiousness = np.zeros(self.model.initial_population.shape)
        # Set all infectious compartments to be equally infectious.
        compartment_infectiousness[infectious_mask] = 1

        for strat in self.model._stratifications:
            for comp_name, adjustments in strat.infectiousness_adjustments.items():
                for stratum, adjustment in adjustments.items():
                    if adjustment:
                        is_overwrite = isinstance(adjustment, Overwrite)
                        adj_value = self._graph_values_static[adjustment.param._graph_key]
                        adj_comps = self.model.get_matching_compartments(
                            comp_name, {strat.name: stratum}
                        )
                        for c in adj_comps:
                            if is_overwrite:
                                compartment_infectiousness[c.idx] = adj_value
                            else:
                                orig_value = compartment_infectiousness[c.idx]
                                compartment_infectiousness[c.idx] = adj_value * orig_value

        if "strain" in self.model.stratifications:
            # FIXME: If there are multiple strains, but one of them is _DEFAULT_DISEASE_STRAIN
            # there will almost certainly be incorrect masks applied
            # Filter out all values that are not in the given strain.
            strain_mask = np.zeros(self.model.initial_population.shape)
            for idx, compartment in enumerate(self.model.compartments):
                if compartment.has_stratum("strain", strain):
                    strain_mask[idx] = 1

            compartment_infectiousness *= strain_mask

        return compartment_infectiousness

    def _precompute_flow_maps(self):
        """Build fast-access arrays of flow indices"""
        f_pos_map = []
        f_neg_map = []
        for i, f in self._iter_non_function_flows:
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

        model_variables = {
            "time": time,
            "compartment_values": compartment_values,
            "computed_values": {},
        }

        self._graph_values_timestep = self.run_graph_ts(
            parameters=self.parameters, model_variables=model_variables
        )

        # Find the effective infectious population for the force of infection (FoI) calculations.
        self._mixing_matrix = mixing_matrix = self._graph_values_timestep["mixing_matrix"]
        self._calculate_strain_infection_values(compartment_values, mixing_matrix)

    def _get_mixing_matrix(self, t):
        # FIXME: Only used by tests, should never be called in real code
        self.prepare_to_run()
        self._prepare_time_step(t, self.model.initial_population)
        return self._mixing_matrix

    def get_flow_weights(self):
        """Collate weights for all model flows at the current timestep"""

        flow_weights = np.zeros(len(self.model._flows))

        for i, f in enumerate(self.model._flows):
            flow_weights[i] = self._graph_values_timestep[f._graph_key]

        return flow_weights

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

    def _get_force_idx(self, source: Compartment):
        """
        Returns the index of the source compartment in the infection multiplier vector.
        """
        return self._category_lookup[source.idx]

    def _get_timestep_deaths(self, *args, **kwargs) -> float:
        assert self._timestep_deaths is not None, "Total deaths has not been set."
        return self._timestep_deaths

    def _get_infection_multiplier_indices(
        self, source: Compartment, dest: Compartment
    ) -> Tuple[str, int]:
        """Return indices for infection frequency lookups"""
        idx = self._get_force_idx(source)
        strain = dest.strata.get("strain", self.model._DEFAULT_DISEASE_STRAIN)
        return idx, strain

    def _get_infection_frequency_multiplier(self, source: Compartment, dest: Compartment) -> float:
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection frequency' calculation.
        """
        idx, strain = self._get_infection_multiplier_indices(source, dest)
        return self._infection_frequency[strain][idx]

    def _get_infection_density_multiplier(self, source: Compartment, dest: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection density' calculation.
        """
        idx, strain = self._get_infection_multiplier_indices(source, dest)
        return self._infection_density[strain][idx]

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

        self._prepare_time_step(time, compartment_vals)

        computed_values = self._graph_values_timestep["computed_values"]

        flow_weights = self.get_flow_weights()

        # These will be filled in afterwards
        populations = compartment_vals[self.population_idx]
        # Update for special cases (population-independent and CrudeBirth)
        if self._has_non_pop_flows:
            populations[self._non_pop_flow_idx] = 1.0
        if self._has_crude_birth:
            populations[self._crude_birth_idx] = find_sum(compartment_vals)

        flow_rates = flow_weights * populations

        # Calculate infection flows
        infect_mul = self._get_infectious_multipliers()
        flow_rates[self.infectious_flow_indices] *= infect_mul

        # ReplacementBirthFlow depends on death flows already being calculated; update here
        if self._has_replacement:
            # Only calculate timestep_deaths if we use replacement, it's expensive...
            self._timestep_deaths = flow_rates[self.death_flow_indices].sum()
            flow_rates[self._replacement_flow_idx] = self._timestep_deaths

        return flow_rates, computed_values

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

        comp_rates = np.zeros(len(comp_vals))
        flow_rates, _ = self._get_flow_rates(comp_vals, time)

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
        comp_vals = clean_compartment_values(compartment_values)
        comp_rates, _ = self._get_rates(comp_vals, time)
        return comp_rates

    def get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Returns the contribution of each flow to compartment rate of change for a given state and
        time.
        """
        comp_vals = clean_compartment_values(compartment_values)
        return self._get_flow_rates(comp_vals, time)

    def calculate_initial_population(self) -> np.ndarray:
        """
        Called to recalculate the initial population from either fixed dictionary, or a dict
        supplied as a parameter
        """
        # FIXME:
        # Work in progress; correctly recalculates non-parameterized
        # populations, but does not include population rebalances etc
        model = self.model

        distribution = model._init_pop_dist
        initial_population = np.zeros_like(model._original_compartment_names, dtype=float)

        # if is_var(distribution, "parameters"):
        #    distribution = self.parameters[distribution.name]
        # elif isinstance(distribution, Function) or isinstance(distribution, ModelParameter):
        #    distribution = get_static_param_value(distribution, parameters)

        if isinstance(distribution, dict):
            for idx, comp in enumerate(model._original_compartment_names):
                pop_key = distribution[comp.name]._graph_key
                pop = self._graph_values_static[pop_key]
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
                        prev_compartment_names, initial_population, self._graph_values_static
                    )
                elif action.action_type == "adjust_pop_split":
                    initial_population = get_rebalanced_population(
                        model, initial_population, self._graph_values_static, **action.kwargs
                    )
        else:
            raise TypeError(
                "Initial population distribution must be a dict",
                distribution,
            )
        model.initial_population = initial_population


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
