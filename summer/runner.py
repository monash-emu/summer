from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numba

import summer.flows as flows

class ModelRunner(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _get_flow_rates(self, compartment_values: np.ndarray, time: float) -> np.ndarray:
        """Returns the contribution of each flow to compartment rate of change for a given state and time.

        Args:
            compartment_values (np.ndarray): Current values of the model compartments
            time (float): Time at which rates are evaluated (expected to be in range of model.times)

        Returns:
            np.ndarray: Array of flow rates (size determined by number of model flows)
        """
        pass

    @abstractmethod
    def _get_compartment_rates(self, compartment_values: np.ndarray, time: float) -> np.ndarray:
        """Interface for the ODE solver: this function is passed to solve_ode func and defines the dynamics of the model.
        

        Args:
            compartment_values (np.ndarray): Current values of the model compartments
            time (float): Time (in model.times coordinates) at which current step is being solved

        Returns:
            np.ndarray: Rates of change for the compartment values for a given state and time.
        """

    @abstractmethod
    def _prepare_to_run(self):
        """Pre-run setup.
        
        Perform any setup/precomputation that can be done prior to model run
        """
        pass

class VectorizedRunner(ModelRunner):
    def __init__(self, model):
        super().__init__(model)

    def _prepare_to_run(self):
        """Do all precomputation here
        """
        self.model._prepare_to_run()
        self.precompute_flow_weights()
        self.precompute_flow_maps()
        self.infectious_flow_indices = [i for i, f in self.model._iter_non_function_flows if isinstance(f, flows.BaseInfectionFlow)]
        self.death_flow_indices = [i for i, f in self.model._iter_non_function_flows if f.is_death_flow]
        self.population_idx = np.array([f.source.idx for i, f in self.model._iter_non_function_flows], dtype=int)

    def precompute_flow_weights(self):
        """Calculate all static flow weights before running, and build indices for time-varying weights
        """
        self.flow_weights = np.zeros(len(self.model._iter_non_function_flows))
        time_varying_flow_weights = []
        time_varying_weight_indices = []
        for i, f in self.model._iter_non_function_flows:
            if f.weight_is_static():
                weight = f.get_weight_value(0)
                self.flow_weights[i] = weight
            else:
                #+++ Not currently used; time-varying weights are generated at runtime
                #param_vals = np.array([f.get_weight_value(t) for t in self.model.times])
                #time_varying_flow_weights.append(param_vals)
                time_varying_weight_indices.append(i)

        self.time_varying_weight_indices = np.array(time_varying_weight_indices, dtype=int)
        #self.time_varying_flow_weights = np.array(time_varying_flow_weights)

    def precompute_flow_maps(self):
        """Build fast-access arrays of flow indices
        """
        f_pos_map = []
        f_neg_map = []
        for i, f in self.model._iter_non_function_flows:
            if f.source:
                f_neg_map.append((i, f.source.idx))
            if f.dest:
                f_pos_map.append((i, f.dest.idx))

        self._pos_flow_map = np.array(f_pos_map, dtype=int)
        self._neg_flow_map = np.array(f_neg_map, dtype=int)

    def apply_precomputed_flow_weights_at_time(self, time: float):
        """Fill flow weights with precomputed values

        Not currently used, but retained for evaluation purposes:
        Use apply_flow_weights_at_time instead

        Args:
            time (float): Time in model.times coordinates
        """
        t = int(time - self.model.times[0])
        self.flow_weights[self.time_varying_weight_indices] = self.time_varying_flow_weights[:,t]

    def apply_flow_weights_at_time(self, time):
        """Calculate time dependent flow weights and insert them into our weights array

        Args:
            time (float): Time in model.times coordinates
        """
        t = time
        for i in self.time_varying_weight_indices:
            f = self.model._flows[i]
            self.flow_weights[i] = f.get_weight_value(t)

    def get_infectious_multipliers(self) -> np.ndarray:
        """Get multipliers for all infectious flows

        Returns:
            np.ndarray: Array of infectiousness multipliers
        """
        multipliers = np.empty(len(self.infectious_flow_indices))
        for i, idx in enumerate(self.infectious_flow_indices):
            f = self.model._flows[idx]
            multipliers[i] = f.find_infectious_multiplier(f.source, f.dest)
        return multipliers

    def get_flow_rates(self, comp_vals: np.ndarray, time: float) -> np.ndarray:
        """Get current flow rates, equivalent to calling get_net_flow on all (non-function) flows

        Args:
            comp_vals (np.ndarray): Compartment values
            time (float): Time in model.times coordinates

        Returns:
            np.ndarray: Array of all (non-function) flow rates
        """
        
        self.apply_flow_weights_at_time(time)

        populations = comp_vals[self.population_idx]
        infect_mul = self.get_infectious_multipliers()
        
        flow_rates = self.flow_weights * populations
        flow_rates[self.infectious_flow_indices] *= infect_mul
        
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
        self.model._prepare_time_step(time, comp_vals)
    
        comp_rates = np.zeros(len(comp_vals))
        flow_rates = self.get_flow_rates(comp_vals, time)

        self.model._total_deaths += flow_rates[self.death_flow_indices].sum()

        accumulate_flow_contributions(flow_rates, comp_rates, self._pos_flow_map, self._neg_flow_map)

        if self.model._iter_function_flows:
            # Evaluate the function flows.
            for flow_idx, flow in self.model._iter_function_flows:
                net_flow = flow.get_net_flow(
                    self.compartments, comp_vals, self._flows, flow_rates, time
                )
                comp_rates[flow.source.idx] -= net_flow
                comp_rates[flow.dest.idx] += net_flow

        return comp_rates, flow_rates

    def _get_compartment_rates(self, compartment_values: np.ndarray, time: float):
        """
        Interface for the ODE solver: this function is passed to solve_ode func and defines the dynamics of the model.
        Returns the rate of change of the compartment values for a given state and time.
        """
        comp_vals = self.model._clean_compartment_values(compartment_values)
        comp_rates, _ = self._get_rates(comp_vals, time)
        return comp_rates
   
    def _get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Returns the contribution of each flow to compartment rate of change for a given state and time.
        """
        comp_vals = self.model._clean_compartment_values(compartment_values)
        _, flow_rates = self._get_rates(comp_vals, time)
        return flow_rates


@numba.jit(nopython=True)
def accumulate_flow_contributions(flow_rates: np.ndarray, comp_rates: np.ndarray, pos_flow_map: np.ndarray, neg_flow_map: np.ndarray):
    """Fast accumulator for summing flow rates into their effects on compartments

    Args:
        flow_rates (np.ndarray): Flow rates to be accumulated
        comp_rates (np.ndarray): Output array of compartment rates
        pos_flow_map (np.ndarray): Array of src (flow), target (compartment) indices
        neg_flow_map (np.ndarray): Array of src (flow), target (compartment) indices
    """
    for src, target in pos_flow_map:
        comp_rates[target] += flow_rates[src]
    for src, target in neg_flow_map:
        comp_rates[target] -= flow_rates[src]

