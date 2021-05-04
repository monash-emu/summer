from abc import ABC, abstractmethod

import numpy as np
from functools import lru_cache

import summer.flows as flows
from summer.compute import binary_matrix_to_sparse_pairs, sparse_pairs_accum
from summer.compartment import Compartment


class ModelRunner(ABC):
    """
    Base runner.
    Child classes may be used by the CompartmentalModel to calculate flow rates.
    """

    def __init__(self, model):
        # Compartmental model
        self.model = model

        # Tracks total deaths per timestep for death-replacement birth flows
        self._timestep_deaths = None

    @abstractmethod
    def get_flow_rates(self, compartment_values: np.ndarray, time: float) -> np.ndarray:
        """
        Returns the contribution of each flow to compartment rate of change for a given state and time.

        Args:
            compartment_values (np.ndarray): Current values of the model compartments
            time (float): Time at which rates are evaluated (expected to be in range of model.times)

        Returns:
            np.ndarray: Array of flow rates (size determined by number of model flows)
        """
        pass

    @abstractmethod
    def get_compartment_rates(self, compartment_values: np.ndarray, time: float) -> np.ndarray:
        """
        Interface for the ODE solver: this function is passed to solve_ode func and defines the dynamics of the model.


        Args:
            compartment_values (np.ndarray): Current values of the model compartments
            time (float): Time (in model.times coordinates) at which current step is being solved

        Returns:
            np.ndarray: Rates of change for the compartment values for a given state and time.
        """
        pass

    def _clean_compartment_values(self, compartment_values: np.ndarray):
        """
        Zero out -ve compartment sizes in flow rate calculations,
        to prevent negative values from messing up the direction of flows.
        We don't expect large -ve values, but there can be small ones due to numerical errors.
        """
        comp_vals = compartment_values.copy()
        zero_mask = comp_vals < 0
        comp_vals[zero_mask] = 0
        return comp_vals

    @abstractmethod
    def prepare_to_run(self):
        """
        Pre-run setup.
        Here we do any calculations/preparation are possible to do before the model runs.
        """
        # Functions will often be called multiple times per timestep, so we cache any time-varying functions.
        # First we find all time varying functions.
        funcs = set()
        for flow in self.model._flows:
            if isinstance(flow, flows.FunctionFlow):
                # Don't cache these, the input arguments cannot be stored in a dict ("non hashable")
                continue
            elif callable(flow.param):
                funcs.add(flow.param)
            for adj in flow.adjustments:
                if adj and callable(adj.param):
                    funcs.add(adj.param)

        # Cache return values to prevent re-computation. This will cause a little memory leak, which is (mostly) fine.
        funcs_cached = {}
        for func in funcs:
            # Floating point return type is 8 bytes, meaning 2**17 values is ~1MB of memory.
            funcs_cached[func] = lru_cache(maxsize=2 ** 17)(func)

        # Finally, replace original functions with cached ones
        for flow in self.model._flows:
            if flow.param in funcs_cached:
                flow.param = funcs_cached[flow.param]
            for adj in flow.adjustments:
                if adj and adj.param in funcs_cached:
                    adj.param = funcs_cached[adj.param]

        # Re-order flows so that they are executed in the correct order:
        #   - exit flows
        #   - entry flows (depends on exit flows for 'replace births' functionality)
        #   - transition flows
        #   - function flows (depend on all other prior flows, since they take flow rate as an input)
        num_flows = len(self.model._flows)
        _exit_flows = [f for f in self.model._flows if issubclass(f.__class__, flows.BaseExitFlow)]
        _entry_flows = [
            f for f in self.model._flows if issubclass(f.__class__, flows.BaseEntryFlow)
        ]
        _transition_flows = [
            f
            for f in self.model._flows
            if issubclass(f.__class__, flows.BaseTransitionFlow)
            and not isinstance(f, flows.FunctionFlow)
        ]
        _function_flows = [f for f in self.model._flows if isinstance(f, flows.FunctionFlow)]
        self._has_function_flows = bool(_function_flows)
        self.model._flows = _exit_flows + _entry_flows + _transition_flows + _function_flows
        # Check we didn't miss any flows
        assert len(self.model._flows) == num_flows, "Some flows were lost when preparing to run."
        # Split flows into two groups for runtime.
        self._iter_function_flows = [
            (i, f) for i, f in enumerate(self.model._flows) if isinstance(f, flows.FunctionFlow)
        ]
        self._iter_non_function_flows = [
            (i, f) for i, f in enumerate(self.model._flows) if not isinstance(f, flows.FunctionFlow)
        ]

        """
        Pre-run calculations to help determine force of infection multiplier at runtime.

        We start with a set of "mixing categories". These categories describe groups of compartments.
        For example, we might have the stratifications age {child, adult} and location {work, home}.
        In this case, the mixing categories would be {child x home, child x work, adult x home, adult x work}.
        Mixing categories are only created when a mixing matrix is supplied during stratification.

        There is a mapping from every compartment to a mixing category.
        This is only true if mixing matrices are supplied only for complete stratifications.
        There are `num_cats` categories and `num_comps` compartments.
        The category matrix is a (num_cats x num_comps) matrix of 0s and 1s, with a 1 when the compartment is in a given category.
        We expect only one category per compartment, but there may be many compartments per category.

        We can multiply the category matrix by the vector of compartment sizes to get the total number of people
        in each mixing category.

        We also create a vector of values in [0, inf) which describes how infectious each compartment is: compartment_infectiousness
        We can use this vector plus the compartment sizes to get the 'effective' number of infectious people per compartment.

        We can use the 'effective infectious' compartment sizes, plus the mixing category matrix
        to get the infected population per mixing category.

        Now that we know:
            - the total population per category
            - the infected population per category
            - the inter-category mixing coefficients (mixing matrix)

        We can calculate infection density or infection frequency transition flows for each category.
        Finally, at runtime, we can lookup which category a given compartment is in and look up its infectious multiplier (density or frequency).
        """
        # Find out the relative infectiousness of each compartment, for each strain.
        # If no strains have been created, we assume a default strain name.
        self._compartment_infectiousness = {
            strain_name: self._get_compartment_infectiousness_for_strain(strain_name)
            for strain_name in self.model._disease_strains
        }

        # Create a matrix that tracks which categories each compartment is in.
        # A matrix with size (num_cats x num_comps).
        # This is a very sparse static matrix, and there's almost certainly a much
        # faster way of using it than naive matrix multiplication
        num_comps = len(self.model.compartments)
        self.num_categories = len(self.model._mixing_categories)
        self._category_lookup = {}  # Map compartments to categories.
        self._category_matrix = np.zeros((self.num_categories, num_comps))
        for i, category in enumerate(self.model._mixing_categories):
            for j, comp in enumerate(self.model.compartments):
                if all(comp.has_stratum(k, v) for k, v in category.items()):
                    self._category_matrix[i][j] = 1
                    self._category_lookup[j] = i
        self._compartment_category_map = binary_matrix_to_sparse_pairs(self._category_matrix)

    def _get_compartment_infectiousness_for_strain(self, strain: str):
        """
        Returns a vector of floats, each representing the relative infectiousness of each compartment.
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

        # Apply infectiousness adjustments.
        for idx, comp in enumerate(self.model.compartments):
            inf_value = compartment_infectiousness[idx]
            for strat in self.model._stratifications:
                for comp_name, adjustments in strat.infectiousness_adjustments.items():
                    for stratum, adjustment in adjustments.items():
                        should_apply_adjustment = adjustment and comp.is_match(
                            comp_name, {strat.name: stratum}
                        )
                        if should_apply_adjustment:
                            # Cannot use time-varying functions for infectiousness adjustments,
                            # because this is calculated before the model starts running.
                            inf_value = adjustment.get_new_value(inf_value, None)

            compartment_infectiousness[idx] = inf_value

        if strain != self.model._DEFAULT_DISEASE_STRAIN:
            # Filter out all values that are not in the given strain.
            strain_mask = np.zeros(self.model.initial_population.shape)
            for idx, compartment in enumerate(self.model.compartments):
                if compartment.has_stratum("strain", strain):
                    strain_mask[idx] = 1

            compartment_infectiousness *= strain_mask

        return compartment_infectiousness

    def _calculate_strain_infection_values(
        self, compartment_values: np.ndarray, mixing_matrix: np.ndarray
    ):
        num_cats = self.num_categories
        # Calculate total number of people per category (for FoI).
        # A vector with size (num_cats).
        self._category_populations = sparse_pairs_accum(
            self._compartment_category_map, compartment_values, num_cats
        )

        # Calculate infectious populations for each strain.
        # Infection density/frequency is the infectious multiplier for each mixing category, calculated for each strain.
        self._infection_density = {}
        self._infection_frequency = {}
        for strain in self.model._disease_strains:
            strain_compartment_infectiousness = self._compartment_infectiousness[strain]

            # Calculate the effective infectious people for each category, including adjustment factors.
            # Returns a vector with size (num_cats x 1).
            infected_values = compartment_values * strain_compartment_infectiousness
            infectious_populations = sparse_pairs_accum(
                self._compartment_category_map, infected_values, num_cats
            )
            self._infection_density[strain] = np.matmul(mixing_matrix, infectious_populations)

            # Calculate the effective infectious person frequency for each category, including adjustment factors.
            # A vector with size (num_cats x 1).
            category_prevalence = infectious_populations / self._category_populations
            self._infection_frequency[strain] = np.matmul(mixing_matrix, category_prevalence)

    def _get_mixing_matrix(self, time: float) -> np.ndarray:
        """
        Returns the final mixing matrix for a given time.
        """
        mixing_matrix = self.model._DEFAULT_MIXING_MATRIX
        for mm_func in self.model._mixing_matrices:
            # Assume each mixing matrix is either an np.ndarray or a function of time that returns one.
            mm = mm_func(time) if callable(mm_func) else mm_func
            # Get Kronecker product of old and new mixing matrices.
            mixing_matrix = np.kron(mixing_matrix, mm)

        return mixing_matrix

    def _get_force_idx(self, source: Compartment):
        """
        Returns the index of the source compartment in the infection multiplier vector.
        """
        return self._category_lookup[source.idx]

    def _get_timestep_deaths(self, *args, **kwargs) -> float:
        assert self._timestep_deaths is not None, "Total deaths has not been set."
        return self._timestep_deaths

    def _get_infection_frequency_multiplier(self, source: Compartment, dest: Compartment) -> float:
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection frequency' calculation.
        """
        idx = self._get_force_idx(source)
        strain = dest.strata.get("strain", self.model._DEFAULT_DISEASE_STRAIN)
        return self._infection_frequency[strain][idx]

    def _get_infection_density_multiplier(self, source: Compartment, dest: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection density' calculation.
        """
        idx = self._get_force_idx(source)
        strain = dest.strata.get("strain", self.model._DEFAULT_DISEASE_STRAIN)
        return self._infection_density[strain][idx]
