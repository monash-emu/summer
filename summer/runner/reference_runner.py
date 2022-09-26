from typing import Tuple

import numpy as np

from .model_runner import ModelRunner


class ReferenceRunner(ModelRunner):
    """
    A less optimized, but easier to understand model runner.
    """

    def __init__(self, model):
        super().__init__(model)

    def prepare_structural(self):
        return super().prepare_structural()

    def prepare_dynamic(self, parameters: dict = None):
        return super().prepare_dynamic(parameters)

    def get_compartment_rates(self, compartment_values: np.ndarray, time: float):
        """
        Interface for the ODE solver: this function is passed to solve_ode func and defines
        the dynamics of the model.
        Returns the rate of change of the compartment values for a given state and time.
        """
        comp_vals = self._clean_compartment_values(compartment_values)
        comp_rates, _ = self._get_rates(comp_vals, time)
        return comp_rates

    def get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Returns the contribution of each flow to compartment rate of change for a given
        state and time.
        """
        comp_vals = self._clean_compartment_values(compartment_values)
        _, flow_rates = self._get_rates(comp_vals, time)
        return flow_rates

    def _prepare_time_step(self, time: float, compartment_values: np.ndarray):
        """
        Pre-timestep setup. This should be run before `_get_rates`.
        Here we set up any stateful updates that need to happen before we get the flow rates.
        """
        # Prepare total deaths for tracking deaths.
        self._timestep_deaths = 0

        # Find the effective infectious population for the force of infection (FoI) calculations.
        mixing_matrix = self._get_mixing_matrix(time)

        # Calculate infection frequency/density for all disease strains
        self._calculate_strain_infection_values(compartment_values, mixing_matrix)

    def _get_rates(
        self, compartment_vals: np.ndarray, time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates inter-compartmental flow rates for a given state and time, including:
            - entry: flows of people into the system
            - exit: flows of people leaving of the system, and;
            - transition: flows of people between compartments

        Args:
            comp_vals: The current state of the model compartments (ie. number of people)
            time: The current time

        Returns:
            comp_rates: Rate of change of compartments
            flow_rates: Contribution of each flow to compartment rate of change

        """
        self._prepare_time_step(time, compartment_vals)
        # Track each flow's flow-rates at this point in time for function flows.
        flow_rates = np.zeros(len(self.model._flows))
        # Track the rate of change of compartments for the ODE solver.
        comp_rates = np.zeros(len(compartment_vals))

        computed_values = self._calc_computed_values(compartment_vals, time)

        # Find the flow rate for each flow.
        for flow_idx, flow in self._iter_non_function_flows:
            # Evaluate all the flows that are not function flows.
            net_flow = flow.get_net_flow(compartment_vals, computed_values, time, self.parameters)
            flow_rates[flow_idx] = net_flow
            if flow.source:
                comp_rates[flow.source.idx] -= net_flow
            if flow.dest:
                comp_rates[flow.dest.idx] += net_flow
            if flow.is_death_flow:
                # Track total deaths for any later birth replacement flows.
                self._timestep_deaths += net_flow

        if self._iter_function_flows:
            # Evaluate the function flows.
            for flow_idx, flow in self._iter_function_flows:
                net_flow = flow.get_net_flow(
                    self.model.compartments,
                    compartment_vals,
                    self.model._flows,
                    flow_rates,
                    computed_values,
                    time,
                )
                flow_rates[flow_idx] = net_flow
                comp_rates[flow.source.idx] -= net_flow
                comp_rates[flow.dest.idx] += net_flow

        return comp_rates, flow_rates
