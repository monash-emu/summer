"""
This module contains the main disease modelling class.
"""
import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple

import networkx
import numpy as np

import summer.flows as flows
from summer import stochastic
from summer.adjust import FlowParam
from summer.compartment import Compartment
from summer.derived_outputs import DerivedOutputRequest, calculate_derived_outputs
from summer.runner import ReferenceRunner, VectorizedRunner
from summer.solver import SolverType, solve_ode
from summer.stratification import Stratification
from summer.utils import get_scenario_start_index

logger = logging.getLogger()

FlowRateFunction = Callable[[List[Compartment], np.ndarray, np.ndarray, float], np.ndarray]


class BackendType:
    REFERENCE = "reference"
    VECTORIZED = "vectorized"


class CompartmentalModel:
    """
    A compartmental disease model

    This model defines a set of compartments which each contain a population.
    Disease dynamics are defined by a set of flows which link the compartments together.
    The model is run over a period of time, starting from some initial conditions to predict the future state of a disease.

    Args:
        times: The start and end times. ***
        compartments: The compartments to simulate.
        infectious_compartments: The compartments which are counted as infectious.
        time_step (optional): The timesteps to return results for. This request does not affect the ODE solver, but is used for the stochastic solver. Defaults to ``1``.

    Attributes:
        times (np.ndarray): The times that the model will simulate.
        compartments (List[Compartment]): The model's compartments.
        initial_population (np.ndarray): The model's starting population. The indices of this
            array match to the ``compartments`` attribute. This is zero by default, but should be set with the ``set_initial_population`` method.
        outputs (np.ndarray): The values of each compartment for each requested timestep. For ``C`` compartments and
            ``T`` timesteps this will be a ``TxC`` matrix. The column indices of this array will match up with ``compartments`` and the row indices will match up with ``times``.
        derived_outputs (Dict[str, np.ndarray]): Additional results that are calculated from ``outputs`` for each timestep.


    """

    _DEFAULT_DISEASE_STRAIN = "default"
    _DEFAULT_MIXING_MATRIX = np.array([[1]])
    _DEFAULT_BACKEND = BackendType.VECTORIZED

    def __init__(
        self,
        times: Tuple[int, int],
        compartments: List[str],
        infectious_compartments: List[str],
        timestep: float = 1.0,
    ):
        start_t, end_t = times
        assert end_t > start_t, "End time must be greater than start time"
        time_period = end_t - start_t
        num_steps = 1 + (time_period / timestep)
        msg = f"Time step {timestep} must be less than time period {time_period}"
        assert num_steps >= 1, msg
        msg = f"Time step {timestep} must be a factor of time period {time_period}"
        assert num_steps % 1 == 0, msg
        self.times = np.linspace(start_t, end_t, num=int(num_steps))
        self.timestep = timestep

        error_msg = "Infectious compartments must be a subset of compartments"
        assert all(n in compartments for n in infectious_compartments), error_msg
        self.compartments = [Compartment(n) for n in compartments]
        self._infectious_compartments = [Compartment(n) for n in infectious_compartments]
        self.initial_population = np.zeros_like(self.compartments, dtype=np.float)
        # Keeps track of original, pre-stratified compartment names.
        self._original_compartment_names = [Compartment.deserialize(n) for n in compartments]
        # Keeps track of Stratifications that have been applied.
        self._stratifications = []
        # Flows to be applied to the model compartments
        self._flows = []

        # The results calculated using the model: no outputs exist until the model has been run.
        self.outputs = None
        self.derived_outputs = None
        # Track 'derived output' requests in a dictionary.
        self._derived_output_requests = {}
        # Track 'derived output' request dependencies in a directed acylic graph (DAG).
        self._derived_output_graph = networkx.DiGraph()
        # Whitelist of 'derived outputs' to evaluate
        self._derived_outputs_whitelist = []

        # Init baseline model to None; can be set via set_baseline if running as a scenario
        self._baseline = None

        # Mixing matrices: a list of square arrays, or functions, used to calculate force of infection.
        self._mixing_matrices = []
        # Mixing categories: a list of dicts that knows the strata required to match a row in the mixing matrix.
        self._mixing_categories = [{}]
        # Strains: a list of the different sub-categories ('strains') of the disease that we are modelling.
        self._disease_strains = [self._DEFAULT_DISEASE_STRAIN]

        self._update_compartment_indices()

    def _update_compartment_indices(self):
        """
        Update the mapping of compartment name to idx for quicker lookups.
        """
        compartment_idx_lookup = {}
        # Update the mapping of compartment name to idx for quicker lookups.
        for idx, c in enumerate(self.compartments):
            c.idx = idx
            compartment_idx_lookup[c] = idx

        for flow in self._flows:
            flow.update_compartment_indices(compartment_idx_lookup)

    def set_initial_population(self, distribution: Dict[str, float]):
        """
        Sets the initial population of the model, which is zero by default.

        Args:
            distribution: A map of populations to be assigned to compartments.

        """
        error_msg = "Cannot set initial population after the model has been stratified"
        assert not self._stratifications, error_msg
        for idx, comp in enumerate(self.compartments):
            pop = distribution.get(comp.name, 0)
            assert pop >= 0, f"Population for {comp.name} cannot be negative: {pop}"
            self.initial_population[idx] = pop

    """
    Adding flows
    """

    def add_crude_birth_flow(
        self,
        name: str,
        birth_rate: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a crude birth rate flow to the model.
        The number of births will be determined by the product of the birth rate and total population.

        Args:
            name: The name of the new flow.
            birth_rate: The fractional crude birth rate per timestep.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._validate_param(name, birth_rate)
        is_already_birth_flow = any(
            [
                type(f) is flows.CrudeBirthFlow or type(f) is flows.ReplacementBirthFlow
                for f in self._flows
            ]
        )
        if is_already_birth_flow:
            msg = "There is already a birth flow in this model, cannot add a second."
            raise ValueError(msg)

        self._add_entry_flow(
            flows.CrudeBirthFlow,
            name,
            birth_rate,
            dest,
            dest_strata,
            expected_flow_count,
        )

    def add_replacement_birth_flow(
        self,
        name: str,
        dest: str,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a death-replacing birth flow to the model.
        The number of births will replace the total number of deaths each year

        Args:
            name: The name of the new flow.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        # Only allow a single replacement flow to be added to the model.
        is_already_birth_flow = any(
            [
                type(f) is flows.CrudeBirthFlow or type(f) is flows.ReplacementBirthFlow
                for f in self._flows
            ]
        )
        if is_already_birth_flow:
            msg = "There is already a birth flow in this model, cannot add a second."
            raise ValueError(msg)

        self._add_entry_flow(
            flows.ReplacementBirthFlow,
            name,
            self._get_timestep_deaths,
            dest,
            dest_strata,
            expected_flow_count,
        )

    def add_importation_flow(
        self,
        name: str,
        num_imported: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds an importation flow to the model, where people enter the destination compartment from outside the system.
        The number of people imported per timestep is completely determined by ``num_imported``.

        Args:
            name: The name of the new flow.
            num_imported: The number of people arriving per timestep.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._validate_param(name, num_imported)
        self._add_entry_flow(
            flows.ImportFlow, name, num_imported, dest, dest_strata, expected_flow_count
        )

    def _add_entry_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
    ):
        dest_strata = dest_strata or {}
        dest_comps = [c for c in self.compartments if c.is_match(dest, dest_strata)]
        new_flows = []
        for dest_comp in dest_comps:
            flow = flow_cls(name, dest_comp, param)
            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows
        self._update_compartment_indices()

    def add_death_flow(
        self,
        name: str,
        death_rate: FlowParam,
        source: str,
        source_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow where people die and leave the compartment, reducing the total population.

        Args:
            name: The name of the new flow.
            death_rate: The fractional death rate per timestep.
            source: The name of the source compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_exit_flow(
            flows.DeathFlow,
            name,
            death_rate,
            source,
            source_strata,
            expected_flow_count,
        )

    def add_universal_death_flows(self, base_name: str, death_rate: FlowParam):
        """
        Adds a universal death rate flow to every compartment in the model.
        The number of deaths per compartment will be determined by the product of
        the death rate and the compartment population.

        The base name will be used to create the name of each flow. For example a
        base name of "universal_death" applied to the "S" compartment will result in a flow called
        "universal_death_for_S".

        Args:
            base_name: The base name for each new flow.
            death_rate: The fractional death rate per timestep.

        Returns:
            List[str]: The names of the flows added.

        """
        # Only allow a single universal death flow with a given name to be added to the model.
        is_already_used = any([f.name.startswith(base_name) for f in self._flows])
        if is_already_used:
            msg = f"There is already a universal death flow called '{base_name}' in this model, cannot add a second."
            raise ValueError(msg)

        flow_names = []
        for comp_name in self._original_compartment_names:
            flow_name = f"{base_name}_for_{comp_name}"
            flow_names.append(flow_name)
            self._add_exit_flow(
                flows.DeathFlow,
                flow_name,
                death_rate,
                comp_name,
                source_strata={},
                expected_flow_count=None,
            )

        return flow_names

    def _add_exit_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        source: str,
        source_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
    ):
        source_strata = source_strata or {}
        self._validate_param(name, param)
        source_comps = [c for c in self.compartments if c.is_match(source, source_strata)]
        new_flows = []
        for source_comp in source_comps:
            flow = flow_cls(name, source_comp, param)
            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows
        self._update_compartment_indices()

    def add_infection_frequency_flow(
        self,
        name: str,
        contact_rate: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow that infects people using an "infection frequency" contact rate, which is
        when the infectious multiplier is determined by the proportion of infectious people to the total population.

        Args:
            name: The name of the new flow.
            contact_rate: The effective contact rate.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.InfectionFrequencyFlow,
            name,
            contact_rate,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
            find_infectious_multiplier=self._get_infection_frequency_multiplier,
        )

    def add_infection_density_flow(
        self,
        name: str,
        contact_rate: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow that infects people using an "infection density" contact rate, which is
        when the infectious multiplier is determined by the number of infectious people.

        Args:
            name: The name of the new flow.
            contact_rate: The contact rate.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.InfectionDensityFlow,
            name,
            contact_rate,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
            find_infectious_multiplier=self._get_infection_density_multiplier,
        )

    def add_transition_flow(
        self,
        name: str,
        fractional_rate: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds a flow transfers people from a source to a destination based on the population of the source
        compartment and the fractional flow rate.

        Args:
            name: The name of the new flow.
            fractional_rate: The fraction of people that transfer per timestep.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.TransitionFlow,
            name,
            fractional_rate,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
        )

    def add_function_flow(
        self,
        name: str,
        flow_rate_func: FlowRateFunction,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        A flow that transfers people from a source to a destination based on a user-defined function.
        This can be used to define more complex flows if required. See `flows.FunctionFlow` for more details on the arguments to the function.

        Args:
            name: The name of the new flow.
            flow_rate_func:  A function that returns the flow rate, before adjustments.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are created.

        """
        self._add_transition_flow(
            flows.FunctionFlow,
            name,
            flow_rate_func,
            source,
            dest,
            source_strata,
            dest_strata,
            expected_flow_count,
        )

    def _add_transition_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        source: str,
        dest: str,
        source_strata: Optional[Dict[str, str]],
        dest_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
        find_infectious_multiplier=None,
    ):
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}
        if flow_cls is not flows.FunctionFlow:
            # Non standard param for FunctionFlow so cannot use this validation method.
            self._validate_param(name, param)

        dest_comps = [c for c in self.compartments if c.is_match(dest, dest_strata)]
        source_comps = [c for c in self.compartments if c.is_match(source, source_strata)]
        num_dest = len(dest_comps)
        num_source = len(dest_comps)
        msg = f"Expected equal number of source and dest compartments, but got {num_source} source and {num_dest} dest."
        assert num_dest == num_source, msg
        new_flows = []
        for source_comp, dest_comp in zip(source_comps, dest_comps):
            if find_infectious_multiplier:
                flow = flow_cls(
                    name,
                    source_comp,
                    dest_comp,
                    param,
                    find_infectious_multiplier=find_infectious_multiplier,
                )
            else:
                flow = flow_cls(name, source_comp, dest_comp, param)

            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows
        self._update_compartment_indices()

    def _validate_param(self, flow_name: str, param: FlowParam):
        """
        Ensure that the supplied parameter produces sensible results for all timesteps.
        """
        is_all_positive = (
            all(map(lambda t: param(t) >= 0, self.times)) if callable(param) else param >= 0
        )
        error_msg = f"Parameter for {flow_name} must be >= 0 for all timesteps: {param}"
        assert is_all_positive, error_msg

    @staticmethod
    def _validate_expected_flow_count(
        expected_count: Optional[int], new_flows: List[flows.BaseFlow]
    ):
        """
        Ensure the number of new flows created is the expected amount
        """
        if expected_count is not None:
            # Check that we added the expected number of flows.
            actual_count = len(new_flows)
            msg = f"Expected to add {expected_count} flows but added {actual_count}"
            assert actual_count == expected_count, msg

    """
    Stratifying the model
    """

    def stratify_with(self, strat: Stratification):
        """
        Apply the stratification to the model's flows and compartments.

        Args:
            strat: The stratification to apply.

        """
        # Validate flow adjustments
        flow_names = [f.name for f in self._flows]
        for n in strat.flow_adjustments.keys():
            msg = f"Flow adjustment for '{n}' refers to a flow that is not present in the model."
            assert n in flow_names, msg

        # Validate infectiousness adjustments.
        msg = "All stratification infectiousness adjustments must refer to a compartment that is present in model."
        assert all(
            [c in self._original_compartment_names for c in strat.infectiousness_adjustments.keys()]
        ), msg

        if strat.mixing_matrix is not None:
            # Add this stratification's mixing matrix if a new one is provided.
            assert not strat.is_strain(), "Strains cannot have a mixing matrix."
            # Only allow mixing matrix to be supplied if there is a complete stratification.
            msg = "Mixing matrices only allowed for full stratification."
            assert strat.compartments == self._original_compartment_names, msg
            self._mixing_matrices.append(strat.mixing_matrix)
            # Update mixing categories for force of infection calculation.
            old_mixing_categories = self._mixing_categories
            self._mixing_categories = []
            for mc in old_mixing_categories:
                for stratum in strat.strata:
                    self._mixing_categories.append({**mc, strat.name: stratum})

        if strat.is_strain():
            # Track disease strain names, overriding default values.
            msg = "An infection strain stratification has already been applied, cannot use this more than once."
            assert not any([s.is_strain() for s in self._stratifications]), msg
            self._disease_strains = strat.strata

        # Stratify compartments, split according to split_proportions
        prev_compartment_names = copy.copy(self.compartments)
        self.compartments = strat._stratify_compartments(self.compartments)
        self.initial_population = strat._stratify_compartment_values(
            prev_compartment_names, self.initial_population
        )

        # Stratify flows
        prev_flows = self._flows
        self._flows = []
        for flow in prev_flows:
            self._flows += flow.stratify(strat)

        # Update the mapping of compartment name to idx for quicker lookups.
        self._update_compartment_indices()

        if strat.is_ageing():
            msg = "Age stratification can only be applied once"
            assert not any([s.is_ageing() for s in self._stratifications]), msg

            # Create inter-compartmental flows for ageing from one stratum to the next.
            # The ageing rate is proportional to the width of the age bracket.
            # It's assumed that both ages and model timesteps are in years.
            ages = list(sorted(map(int, strat.strata)))

            # Only allow age stratification to be applied with complete stratifications, because everyone has an age.
            msg = "Mixing matrices only allowed for full stratification."
            assert strat.compartments == self._original_compartment_names, msg

            for age_idx in range(len(ages) - 1):
                start_age = int(ages[age_idx])
                end_age = int(ages[age_idx + 1])
                for comp in prev_compartment_names:

                    source = comp.stratify(strat.name, str(start_age))
                    dest = comp.stratify(strat.name, str(end_age))
                    ageing_rate = 1.0 / (end_age - start_age)
                    self.add_transition_flow(
                        name=f"ageing_{source}_to_{dest}",
                        fractional_rate=ageing_rate,
                        source=source.name,
                        dest=dest.name,
                        source_strata=source.strata,
                        dest_strata=dest.strata,
                        expected_flow_count=1,
                    )

        self._stratifications.append(strat)

    """
    Running the model
    """

    def run(
        self,
        solver: str = SolverType.SOLVE_IVP,
        backend: str = _DEFAULT_BACKEND,
        backend_args: dict = None,
        **kwargs,
    ):
        """
        Runs the model over the provided time span, calculating the outputs and the derived outputs.
        The model calculates the outputs by solving an ODE which is defined by the initial population and the inter-compartmental flows.

        **Note**: The default ODE solver used to produce the model's outputs does not necessarily evaluate every requested timestep. This adaptive
        solver can skip over times, or double back when trying to characterize the ODE. The final results are produced by interpolating the
        solution produced by the ODE solver. This means that model dynamics that only occur in short time periods may not be reflected in the outputs.

        Args:
            solver (optional): The ODE solver to use, defaults to SciPy's IVP solver.
            **kwargs (optional): Extra arguments to supplied to the solver, see ``summer.solver`` for details.

        """

        self._set_backend(backend, backend_args)
        self._backend.prepare_to_run()

        if solver == SolverType.STOCHASTIC:
            # Run the model in 'stochastic mode'.
            seed = kwargs.get("seed")
            self._solve_stochastic(seed)
        else:
            # Run the model as a deterministic ODE
            self._solve_ode(solver, kwargs)

        # Calculate any requested derived outputs, based on the calculated compartment sizes.
        self.derived_outputs = self._calculate_derived_outputs()

    def _set_backend(self, backend: str, backend_args: dict = None):
        backend_args = backend_args or {}
        if backend == BackendType.REFERENCE:
            self._backend = ReferenceRunner(self, **backend_args)
        elif backend == BackendType.VECTORIZED:
            self._backend = VectorizedRunner(self, **backend_args)
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

    def run_stochastic(
        self,
        seed: Optional[int] = None,
        backend: str = _DEFAULT_BACKEND,
        backend_args: dict = None,
    ):
        """
        Runs the model over the provided time span, calculating the outputs and the derived outputs.
        Uses an stochastic interpretation of flow rates.
        """
        self.run(
            solver=SolverType.STOCHASTIC, seed=seed, backend=backend, backend_args=backend_args
        )

    def _solve_ode(self, solver, solver_args: dict):
        """
        Runs the model over the provided time span, calculating the outputs.
        Uses an ODE interpretation of flow rates.
        """

        # Calculate the outputs (compartment sizes) by solving the ODE defined by _get_compartment_rates().
        self.outputs = solve_ode(
            solver,
            self._backend.get_compartment_rates,
            self.initial_population,
            self.times,
            solver_args,
        )

    def _solve_stochastic(self, seed: Optional[int] = None):
        """
        Runs the model over the provided time span, calculating the outputs.
        This method is stochastic: each run may produce different results.
        A random seed (eg. 12345, 1337) can be provided to ensure the same results are produced.

        With this approach we represent a discrete number of people, unlike in the ODE solver, there
        is way to have of 'half a person' or a compartment size of 101.23. Both compartment sizes and
        flows are discrete.

        We use an stochastic interpretation of flow rates.
        The flow rates from _get_rates() are used to calculate the probability of an person
            - entering the model (births, imports)
            - leaving the model (deaths)
            - moving between compartments (transitions, infections)

        There are two sampling methods used:
            - Transition and exit flows are sampled from an exponential distribution using a multinomial
            - Entry flows are sampled froma  Poisson distribution

        The main difference is that there is no 'source' compartment for entry flows.

        See here for more detail on the underlying theory and how this approach can be used:

            http://summerepi.com/examples/5-stochastic-solver.html#How-it-works

        There are three main steps to this method:
            - 1. calculate the flow rates
            - 2. split the flow rates into entry or transition flows
            - 3. calculate and sample the probabilities given the flow rates

        """
        self.outputs = np.zeros((len(self.times), len(self.initial_population)), dtype=np.int)
        self.outputs[0] = self.initial_population

        # Create an array that maps flows to the source and destination compartments.
        # This is done because numba like ndarrays, and numba is fast.
        flow_map = stochastic.build_flow_map(self._flows)

        # Evaluate the model: calculate compartment sizes for each timestep.
        for time_idx, time in enumerate(self.times):
            if time_idx == 0:
                # Skip time zero, use initial conditions.
                continue

            # Calculate the flow rates at this timestep.
            # These describe the rate (people/timeunit) at which people follow the flow.
            # Later we will convert these to probabilities.
            comp_vals = self._backend._clean_compartment_values(self.outputs[time_idx - 1])
            flow_rates = self._backend.get_flow_rates(comp_vals, time)

            # We split the calculatd flow rates into entry or {transition, exit} flows,
            # because we handle them separately with different methods.

            # Transition (and exit) flow are stored in a 2D FxC ndarray where f is the flow idx
            # and c is the compartment idx, giving us a matrix of flows rates out of each compartment.
            transition_flow_rates = np.zeros((len(self._flows), len(comp_vals)))

            # Entry flows are stored in 1D C sized ndarray with one element per compartment.
            # Giving us a vector of *net* flow rates into each compartment from outside the system.
            entry_flow_rates = np.zeros_like(comp_vals)

            # Split the flow rates into entry or {transition, exit} flows.
            for flow_idx, flow in enumerate(self._flows):
                if flow.source:
                    # It's an exit or transition flow, which we sample with a multinomial.
                    transition_flow_rates[flow_idx][flow.source.idx] = flow_rates[flow_idx]
                else:
                    # It's an entry flow, which we sample with a Poisson distribution.
                    entry_flow_rates[flow.dest.idx] += flow_rates[flow_idx]

            # Convert flow rates to probabilities, and take a sample using these probabilities,
            # so we end up with the changes to the compartment sizes due to flows over this timestep.
            entry_changes = stochastic.sample_entry_flows(seed, entry_flow_rates, self.timestep)
            transition_changes = stochastic.sample_transistion_flows(
                seed, transition_flow_rates, flow_map, comp_vals, self.timestep
            )

            # Calculate final compartment sizes at this timestep.
            self.outputs[time_idx] = comp_vals + transition_changes + entry_changes

    def _get_infection_frequency_multiplier(self, *args, **kwargs) -> float:
        return self._backend._get_infection_frequency_multiplier(*args, **kwargs)

    def _get_infection_density_multiplier(self, *args, **kwargs):
        return self._backend._get_infection_density_multiplier(*args, **kwargs)

    def _get_timestep_deaths(self, *args, **kwargs):
        return self._backend._get_timestep_deaths(*args, **kwargs)

    """
    Requesting and calculating derived outputs
    """

    def set_derived_outputs_whitelist(self, whitelist: List[str]):
        """
        Request that we should only calculate a subset of the model's derived outputs.
        This can be useful when you only care about some results and you want to cut down on runtime.
        For example, we may only need some derived outputs for calibration, but may need more later when we want to know
        all the dyanmics that the model actually showed.

        Args:
            whitelist: A list of the derived output names to calculate, ignoring all others.

        """
        self._derived_outputs_whitelist = whitelist

    def set_baseline(self, baseline):
        """Set a baseline model to be used for this (scenario) run
        Sets initial population values to the baseline values for this model's start time
        Cumulative and relative outputs will refer to the baseline

        Args:
            baseline (CompartmentalModel): The baseline model to be used as reference
        """
        start_index = get_scenario_start_index(baseline.times, self.times[0])
        init_compartments = baseline.outputs[start_index, :]
        self.initial_population = init_compartments
        self._baseline = baseline

    def _calculate_derived_outputs(self):
        return calculate_derived_outputs(
            requests=self._derived_output_requests,
            graph=self._derived_output_graph,
            outputs=self.outputs,
            times=self.times,
            timestep=self.timestep,
            flows=self._flows,
            compartments=self.compartments,
            get_flow_rates=self._backend.get_flow_rates,
            whitelist=self._derived_outputs_whitelist,
            baseline=self._baseline
        )

    def request_output_for_flow(
        self,
        name: str,
        flow_name: str,
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
        save_results: bool = True,
        raw_results: bool = False,
    ):
        """
        Adds a derived output to the model's results.
        The output will be the value of the requested flow at each timestep.

        Args:
            name: The name of the derived output.
            flow_name: The name of the flow to track.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            save_results (optional): Whether to save or discard the results. Defaults to ``True``.
            raw_results (optional): Whether to use raw interpolated flow rates, or post-process them so that they're more
                represenative of the changes in compartment sizes. Defaults to ``False``.
        """
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        is_flow_exists = any(
            [f.is_match(flow_name, source_strata, dest_strata) for f in self._flows]
        )
        assert is_flow_exists, f"No flow matches: {flow_name} {source_strata} {dest_strata}"
        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.FLOW,
            "flow_name": flow_name,
            "source_strata": source_strata,
            "dest_strata": dest_strata,
            "raw_results": raw_results,
            "save_results": save_results,
        }

    def request_output_for_compartments(
        self,
        name: str,
        compartments: List[str],
        strata: Optional[Dict[str, str]] = None,
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output
        will be the aggregate population of the requested compartments at the at each timestep.

        Args:
            name: The name of the derived output.
            compartments: The name of the compartments to track.
            strata (optional): A whitelist of strata to filter the compartments.
            save_results (optional): Whether to save or discard the results.
        """
        strata = strata or {}
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        is_match_exists = any(
            [any([c.is_match(name, strata) for name in compartments]) for c in self.compartments]
        )
        assert is_match_exists, f"No compartment matches: {compartments} {strata}"
        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.COMPARTMENT,
            "compartments": compartments,
            "strata": strata,
            "save_results": save_results,
        }

    def request_aggregate_output(
        self,
        name: str,
        sources: List[str],
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output will be the aggregate of other derived outputs.

        Args:
            name: The name of the derived output.
            sources: The names of the derived outputs to aggregate.
            save_results (optional): Whether to save or discard the results.

        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        for source in sources:
            assert (
                source in self._derived_output_requests
            ), f"Source {source} has not been requested."
            self._derived_output_graph.add_edge(source, name)

        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.AGGREGATE,
            "sources": sources,
            "save_results": save_results,
        }

    def request_cumulative_output(
        self,
        name: str,
        source: str,
        start_time: int = None,
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output will be the accumulated values of another derived
        output over the model's time period.

        Args:
            name: The name of the derived output.
            source: The name of the derived outputs to accumulate.
            start_time (optional): The time to start accumulating from, defaults to model start time.
            save_results (optional): Whether to save or discard the results.

        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        assert source in self._derived_output_requests, f"Source {source} has not been requested."
        self._derived_output_graph.add_node(name)
        self._derived_output_graph.add_edge(source, name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.CUMULATIVE,
            "source": source,
            "start_time": start_time,
            "save_results": save_results,
        }

    def request_function_output(
        self,
        name: str,
        func: Callable[[np.ndarray], np.ndarray],
        sources: List[str],
        save_results: bool = True,
    ):
        """
        Adds a derived output to the model's results. The output will be the result of a function
        which takes a list of sources as an input.

        Args:
            name: The name of the derived output.
            func: A function used to calculate the derived ouput.
            sources: The derived ouputs to input into the function.
            save_results (optional): Whether to save or discard the results.

        Example:
            Request a function-based derived output:

                model.request_output_for_compartments(
                    compartments=["S", "E", "I", "R"],
                    name="total_population",
                    save_results=False
                )
                model.request_output_for_compartments(
                    compartments=["R"],
                    name="recovered_population",
                    save_results=False
                )

                def calculate_proportion_seropositive(recovered_pop, total_pop):
                    return recovered_pop / total_pop

                model.request_function_output(
                    name="proportion_seropositive",
                    func=calculate_proportion_seropositive,
                    sources=["recovered_population", "total_population"],
                )

        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        for source in sources:
            assert (
                source in self._derived_output_requests
            ), f"Source {source} has not been requested."
            self._derived_output_graph.add_edge(source, name)

        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.FUNCTION,
            "func": func,
            "sources": sources,
            "save_results": save_results,
        }
