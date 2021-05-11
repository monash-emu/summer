from typing import Tuple

import numpy as np

import summer.flows as flows
from summer.compute import (
    accumulate_positive_flow_contributions,
    accumulate_negative_flow_contributions,
)

from .model_runner import ModelRunner


class VectorizedRunner(ModelRunner):
    """
    An optimized, but less accessible model runner.
    """

    def __init__(self, model, precompute_time_flows=False, precompute_mixing=False):
        super().__init__(model)
        self._precompute_time_flows = precompute_time_flows
        self._precompute_mixing = precompute_mixing

    def prepare_to_run(self):
        """Do all precomputation here"""
        super().prepare_to_run()
        
        self.infectious_flow_indices = [
            i for i, f in self._iter_non_function_flows if isinstance(f, flows.BaseInfectionFlow)
        ]
        self.death_flow_indices = [i for i, f in self._iter_non_function_flows if f.is_death_flow]
        
        # Include dummy values in population_idx to account for Entry flows
        non_func_pops = np.array(
            [f.source.idx if f.source else 0 for i, f in self._iter_non_function_flows], dtype=int
        )

        func_pops = np.array(
            [f.source.idx if f.source else 0 for i, f in self._iter_function_flows], dtype=int
        )

        self.population_idx = np.concatenate((non_func_pops, func_pops))

        # Store indices of flows that are not population dependent
        self._non_pop_flow_idx = np.array([i for i, f in self._iter_non_function_flows \
            if (type(f) in (flows.ReplacementBirthFlow, flows.ImportFlow))], dtype=int)
        self._has_non_pop_flows = bool(len(self._non_pop_flow_idx))

        # Crude birth flows use population sum rather than a compartment; store indices here
        self._crude_birth_idx = np.array([i for i, f in self._iter_non_function_flows \
            if type(f) == flows.CrudeBirthFlow], dtype=int)
        self._has_crude_birth = bool(len(self._crude_birth_idx))

        self._has_replacement = False
        # Replacement flows must be calculated after death flows, store indices here
        for i, f in self._iter_non_function_flows:
            if type(f) == flows.ReplacementBirthFlow:
                self._has_replacement = True
                self._replacement_flow_idx = i

        self._precompute_flow_weights()
        self._precompute_flow_maps()

        if self._precompute_mixing:
            self._precompute_mixing_matrices()

    def _precompute_flow_weights(self):
        """Calculate all static flow weights before running, and build indices for time-varying weights"""
        self.flow_weights = np.zeros(len(self.model._flows))
        time_varying_flow_weights = []
        time_varying_weight_indices = []
        for i, f in self._iter_non_function_flows:
            if f.weight_is_static():
                weight = f.get_weight_value(0)
                self.flow_weights[i] = weight
            else:
                if self._precompute_time_flows:
                    param_vals = np.array([f.get_weight_value(t) for t in self.model.times])
                    time_varying_flow_weights.append(param_vals)
                time_varying_weight_indices.append(i)

        self.time_varying_weight_indices = np.array(time_varying_weight_indices, dtype=int)
        self.time_varying_flow_weights = np.array(time_varying_flow_weights)

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

    def _precompute_mixing_matrices(self):
        num_cat = self.num_categories
        self.mixing_matrices = np.empty((len(self.model.times), num_cat, num_cat))
        for i, t in enumerate(self.model.times):
            self.mixing_matrices[i] = super()._get_mixing_matrix(t)

    def _prepare_time_step(self, time: float, compartment_values: np.ndarray):
        """
        Pre-timestep setup. This should be run before `_get_rates`.
        Here we set up any stateful updates that need to happen before we get the flow rates.
        """
        
        # Some flows (e.g birth replacement) expect this value to be defined 
        self._timestep_deaths = 0.

        # Find the effective infectious population for the force of infection (FoI) calculations.
        mixing_matrix = self._get_mixing_matrix(time)

        self._calculate_strain_infection_values(compartment_values, mixing_matrix)

    def _get_mixing_matrix(self, time: float) -> np.ndarray:
        """Thin wrapper to either get the model's mixing matrix, or use our precomputed matrices

        Args:
            time (float): Time in model.times

        Returns:
            np.ndarray: Mixing matrix at time (time)
        """
        if self._precompute_mixing:
            t = int(time - self.model.times[0])
            return self.mixing_matrices[t]
        else:
            return super()._get_mixing_matrix(time)

    def _apply_precomputed_flow_weights_at_time(self, time: float):
        """Fill flow weights with precomputed values

        Not currently used, but retained for evaluation purposes:
        Use apply_flow_weights_at_time instead

        Args:
            time (float): Time in model.times coordinates
        """

        # Test to see if we have any time varying weights
        if len(self.time_varying_flow_weights):
            t = int(time - self.model.times[0])
            self.flow_weights[self.time_varying_weight_indices] = self.time_varying_flow_weights[
                :, t
            ]

    def _apply_flow_weights_at_time(self, time):
        """Calculate time dependent flow weights and insert them into our weights array

        Args:
            time (float): Time in model.times coordinates
        """
        t = time
        for i in self.time_varying_weight_indices:
            f = self.model._flows[i]
            self.flow_weights[i] = f.get_weight_value(t)

    def _get_infectious_multipliers(self) -> np.ndarray:
        """Get multipliers for all infectious flows

        Returns:
            np.ndarray: Array of infectiousness multipliers
        """
        multipliers = np.empty(len(self.infectious_flow_indices))
        for i, idx in enumerate(self.infectious_flow_indices):
            f = self.model._flows[idx]
            multipliers[i] = f.find_infectious_multiplier(f.source, f.dest)
        return multipliers

    def _get_flow_rates(self, comp_vals: np.ndarray, time: float) -> np.ndarray:
        """Get current flow rates, equivalent to calling get_net_flow on all (non-function) flows

        Args:
            comp_vals (np.ndarray): Compartment values
            time (float): Time in model.times coordinates

        Returns:
            np.ndarray: Array of all (non-function) flow rates
        """
        if self._precompute_time_flows:
            self._apply_precomputed_flow_weights_at_time(time)
        else:
            self._apply_flow_weights_at_time(time)

        
        # These will be filled in afterwards
        populations = comp_vals[self.population_idx]
        # Update for special cases (population-independent and CrudeBirth)
        if self._has_non_pop_flows:
            populations[self._non_pop_flow_idx] = 1.0
        if self._has_crude_birth:
            populations[self._crude_birth_idx] = flows._find_sum(comp_vals)

        flow_rates = self.flow_weights * populations
        
        # Calculate infection flows
        infect_mul = self._get_infectious_multipliers()
        flow_rates[self.infectious_flow_indices] *= infect_mul

        self._timestep_deaths = flow_rates[self.death_flow_indices].sum()
        
        # ReplacementBirthFlow depends on death flows already being calculated; update here
        if self._has_replacement:
            flow_rates[self._replacement_flow_idx] = self._timestep_deaths

        if self._iter_function_flows:
            # Evaluate the function flows.
            for flow_idx, flow in self._iter_function_flows:
                net_flow = flow.get_net_flow(
                    self.model.compartments, comp_vals, self.model._flows, flow_rates, time
                )
                flow_rates[flow_idx] = net_flow

        return flow_rates

    def _get_rates(self, comp_vals: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates inter-compartmental flow rates for a given state and time, as well
        as the updated compartment values once these rate deltas have been applied

        Args:
            comp_vals (np.ndarray): The current state of the model compartments (ie. number of people)
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
        Interface for the ODE solver: this function is passed to solve_ode func and defines the dynamics of the model.
        Returns the rate of change of the compartment values for a given state and time.
        """
        comp_vals = self._clean_compartment_values(compartment_values)
        comp_rates, _ = self._get_rates(comp_vals, time)
        return comp_rates

    def get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Returns the contribution of each flow to compartment rate of change for a given state and time.
        """
        comp_vals = self._clean_compartment_values(compartment_values)
        _, flow_rates = self._get_rates(comp_vals, time)
        return flow_rates
