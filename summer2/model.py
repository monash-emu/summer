"""
This module contains the main disease modelling class.
"""
import copy
import logging
from datetime import datetime
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple
from warnings import warn
import itertools

import networkx
import numpy as np
import pandas as pd

from computegraph import ComputeGraph

import summer2.flows as flows
from summer2.adjust import BaseAdjustment, FlowParam, Multiply
from summer2.compartment import Compartment
from summer2.derived_outputs import DerivedOutputRequest, calculate_derived_outputs
from summer2.parameters import params

from summer2.parameters.param_impl import finalize_parameters
from summer2.runner import ModelBackend
from summer2.solver import SolverType, solve_ode
from summer2.stratification import Stratification
from summer2.utils import get_scenario_start_index, ref_times_to_dti, clean_compartment_values
from summer2.population import get_unique_strat_groups, filter_by_strata
from summer2.tracker import ModelBuildTracker, ActionType

logger = logging.getLogger()

FlowRateFunction = Callable[[List[Compartment], np.ndarray, np.ndarray, float], np.ndarray]


class BackendType:
    PYTHON = "python"
    JAX = "jax"


class CompartmentalModel:
    """
    A compartmental disease model

    This model defines a set of compartments which each contain a population.
    Disease dynamics are defined by a set of flows which link the compartments together.
    The model is run over a period of time, starting from some initial conditions to
    predict the future state of a disease.

    Args:
        times: The start and end times. ***
        compartments: The compartments to simulate.
        infectious_compartments: The compartments which are counted as infectious.
        time_step (optional): The timesteps to return results for. This request does not
        affect the ODE solver, but is used for the stochastic solver. Defaults to ``1``.

    Attributes:
        times (np.ndarray): The times that the model will simulate.
        compartments (List[Compartment]): The model's compartments.
        initial_population (np.ndarray): The model's starting population. The indices of this
            array match to the ``compartments`` attribute. This is zero by default, but should
            be set with the ``set_initial_population`` method.
        outputs (np.ndarray): The values of each compartment for each requested timestep.
            For ``C`` compartments and ``T`` timesteps this will be a ``TxC`` matrix.
            The column indices of this array will match up with ``compartments`` and the
            row indices will match up with ``times``.
        derived_outputs (Dict[str, np.ndarray]): Additional results that are calculated from
            ``outputs`` for each timestep.


    """

    _DEFAULT_DISEASE_STRAIN = "default"
    _DEFAULT_MIXING_MATRIX = np.array([[1.0]])
    _DEFAULT_BACKEND = BackendType.JAX

    def __init__(
        self,
        times: Tuple[int, int],
        compartments: List[str],
        infectious_compartments: List[str],
        timestep: float = 1.0,
        ref_date: datetime = None,
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

        # Set the ref_date; the datetime object equivalent to times[0]
        self.ref_date = ref_date

        if isinstance(infectious_compartments, str):
            infectious_compartments = [infectious_compartments]
        error_msg = "Infectious compartments must be a subset of compartments"
        assert all(n in compartments for n in infectious_compartments), error_msg
        self.compartments = [Compartment(n, idx=i) for i, n in enumerate(compartments)]
        for c in self.compartments:
            if c in infectious_compartments:
                c.tags.append("infectious")

        # Compartment name lookup; needs to be present before adding any flows
        self._update_compartment_name_map()

        self._infectious_compartments = [Compartment(n) for n in infectious_compartments]
        self.initial_population = np.zeros_like(self.compartments, dtype=np.float)
        # Keeps track of original, pre-stratified compartment names.
        self._original_compartment_names = [Compartment.deserialize(n) for n in compartments]
        # Keeps track of Stratifications that have been applied.
        self._stratifications = []
        self.stratifications = {}
        # Flows to be applied to the model compartments
        self._flows = []

        # Turn on all runtime assertions by default; can be disabled for performance reasons
        # Setting to False still retains some checking, but turns off the most costly checks
        self.set_validation_enabled(True)
        self._derived_outputs_idx_cache = {}

        # The results calculated using the model: no outputs exist until the model has been run.
        self.outputs = None
        self.derived_outputs = None
        # Track 'derived output' requests in a dictionary.
        self._derived_output_requests = {}
        # Track 'derived output' request dependencies in a directed acylic graph (DAG).
        self._derived_output_graph = networkx.DiGraph()
        # Whitelist of 'derived outputs' to evaluate
        self._derived_outputs_whitelist = []

        # Map of (runtime) computed values
        self._computed_values_graph_dict = {}

        # Init baseline model to None; can be set via set_baseline if running as a scenario
        self._baseline = None

        # Mixing matrices: a list of square arrays, or functions, used to calculate
        # force of infection.
        self._mixing_matrices = []
        # Mixing categories: a list of dicts that knows the strata required to match a row in
        # the mixing matrix.
        self._mixing_categories = [{}]
        # Strains: a list of the different sub-categories ('strains') of the disease that we
        # are modelling.
        self._disease_strains = [self._DEFAULT_DISEASE_STRAIN]

        self._update_compartment_indices()

        # Track the actions we take building this model
        self.tracker = ModelBuildTracker()

        self._defer_actions = True

        self._finalized = False
        self._runner = None

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

    def _assert_not_finalized(self):
        error_msg = "Cannot make changes to model that is already finalized"
        assert not self._finalized, error_msg

    def set_initial_population(self, distribution: Dict[str, float], force=False):
        """
        Sets the initial population of the model, which is zero by default.

        Args:
            distribution: A map of populations to be assigned to compartments.

        """
        if not force:
            self._assert_not_finalized()

            error_msg = "Cannot set initial population after the model has been stratified"
            assert not self._stratifications, error_msg

        assert isinstance(distribution, dict)

        # Make sure we're not supplying any eroneous compartment names
        for k, v in distribution.items():
            assert k in self._original_compartment_names

        self._init_pop_dist = distribution.copy()

        # Ensure dictionary contains all comparments (default to 0 if not supplied)
        for idx, comp in enumerate(self._original_compartment_names):
            if comp not in self._init_pop_dist:
                self._init_pop_dist[comp] = 0

    def finalize(self):
        if not self._finalized:
            finalize_parameters(self)
            self._finalized = True

    def get_object_graph(self, obj) -> ComputeGraph:
        """Return the ComputeGraph for this object

        Args:
            obj: Any summer object with a graph_key
        """
        if isinstance(obj, dict):
            targets = [v._graph_key for v in obj.values()]
        else:
            targets = obj._graph_key
        return self.graph.filter(targets)

    def set_validation_enabled(self, validate: bool):
        """
        Set this to False in order to turn of some (potentially expensive) runtime validation
        E.g. In calibration, leave it enabled for the first iteration, to catch any structural
        model issues, but then disable for subsequent iterations
        """
        self._should_validate = validate

    def _set_derived_outputs_index_cache(self, idx_cache: dict):
        self._derived_outputs_idx_cache = idx_cache

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
        The number of births will be determined by the product of the birth rate
        and total population.

        Args:
            name: The name of the new flow.
            birth_rate: The fractional crude birth rate per timestep.
            dest: The name of the destination compartment.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows
                                            are created.

        """
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
            expected_flow_count (optional): Used to assert that a particular number of flows
                                            are created.

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
            1.0,
            dest,
            dest_strata,
            expected_flow_count,
        )

    def add_importation_flow(
        self,
        name: str,
        num_imported: FlowParam,
        dest: str,
        split_imports: bool,
        dest_strata: Optional[Dict[str, str]] = None,
        expected_flow_count: Optional[int] = None,
    ):
        """
        Adds an importation flow to the model, where people enter the destination compartment from
        outside the system.
        The number of people imported per timestep is completely determined by ``num_imported``.

        Args:
            name: The name of the new flow.
            num_imported: The number of people arriving per timestep.
            dest: The name of the destination compartment.
            split_imports: Whether to split num_imported amongst existing destination
                           compartments (True), or add the full value to each (False)
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are
                                            created.

        """

        dest_strata = dest_strata or {}
        dest_comps = [c for c in self.compartments if c.is_match(dest, dest_strata)]

        if split_imports:
            # Adjust each of the flows so they are split equally between dest_comps
            adjustments = [Multiply(1.0 / len(dest_comps))]
        else:
            # No adjustment - flow to each dest will be num_imported
            adjustments = None

        self._add_entry_flow(
            flows.ImportFlow,
            name,
            num_imported,
            dest,
            dest_strata,
            expected_flow_count,
            adjustments,
        )

    def _add_entry_flow(
        self,
        flow_cls,
        name: str,
        param: FlowParam,
        dest: str,
        dest_strata: Optional[Dict[str, str]],
        expected_flow_count: Optional[int],
        adjustments: List[BaseAdjustment] = None,
    ):
        self._assert_not_finalized()
        dest_strata = dest_strata or {}
        dest_comps = [c for c in self.compartments if c.is_match(dest, dest_strata)]
        new_flows = []
        for dest_comp in dest_comps:
            flow = flow_cls(name, dest_comp, param, adjustments=adjustments)
            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows

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
            expected_flow_count (optional): Used to assert that a particular number of flows are
                                            created.

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
            msg = f"There is already a universal death flow called '{base_name}' in this model, \
                cannot add a second."
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
        self._assert_not_finalized()
        source_strata = source_strata or {}
        source_comps = [c for c in self.compartments if c.is_match(source, source_strata)]
        new_flows = []
        for source_comp in source_comps:
            flow = flow_cls(name, source_comp, param)
            new_flows.append(flow)

        self._validate_expected_flow_count(expected_flow_count, new_flows)
        self._flows += new_flows

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
        when the infectious multiplier is determined by the proportion of infectious people to the
        total population.

        Args:
            name: The name of the new flow.
            contact_rate: The effective contact rate.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are
                                            created.

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
            expected_flow_count (optional): Used to assert that a particular number of flows are
                                            created.

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
        absolute=False,
    ):
        """
        Adds a flow transferring people from a source to a destination based on the population of
        the source compartment and the fractional flow rate.

        Args:
            name: The name of the new flow.
            fractional_rate: The fraction of people that transfer per timestep.
            source: The name of the source compartment.
            dest: The name of the destination compartment.
            source_strata (optional): A whitelist of strata to filter the source compartments.
            dest_strata (optional): A whitelist of strata to filter the destination compartments.
            expected_flow_count (optional): Used to assert that a particular number of flows are
                                            created.

        """
        if absolute:
            self._add_transition_flow(
                flows.AbsoluteFlow,
                name,
                fractional_rate,
                source,
                dest,
                source_strata,
                dest_strata,
                expected_flow_count
            )
        else:
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
        self._assert_not_finalized()
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}

        dest_comps = self.get_matching_compartments(dest, dest_strata)
        source_comps = self.get_matching_compartments(source, source_strata)

        num_dest = len(dest_comps)
        num_source = len(source_comps)
        msg = f"Expected equal number of source and dest compartments, but got {num_source} source \
                and {num_dest} dest."
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
    Methods for searching and indexing flows and compartments
    """

    def _update_compartment_name_map(self):
        names = set([c.name for c in self.compartments])
        name_map = {}
        for n in names:
            name_map[n] = [c for c in self.compartments if c.name == n]
        self._compartment_name_map = name_map

    def get_matching_compartments(self, name: str, strata: dict):
        if isinstance(name, str):
            name_query = self._compartment_name_map[name]
        else:
            # FIXME: Should do better type checking here
            # For now we assume we have some kind of iterable (ie a 'list' of names)
            match_lists = [self._compartment_name_map[n] for n in name]
            name_query = list(itertools.chain.from_iterable(match_lists))

        if not len(strata):
            return name_query
        else:
            _strata = frozenset(strata.items())
            return [c for c in name_query if c._has_strata(_strata)]

    def _get_matching_compartments(self, name: str, strata: frozenset):
        """
        Optimized version of above method that uses frozenset directly
        """
        name_query = self._compartment_name_map[name]

        if not len(strata):
            return name_query
        else:
            return [c for c in name_query if c._has_strata(strata)]

    """
    Stratifying the model
    """

    def stratify_with(self, strat: Stratification):
        """
        Apply the stratification to the model's flows and compartments.

        Args:
            strat: The stratification to apply.

        """

        self._assert_not_finalized()
        # Enable/disable runtime assertions for strat
        strat._validate = self._should_validate

        # Validate flow adjustments
        flow_names = [f.name for f in self._flows]
        for n in strat.flow_adjustments.keys():
            msg = f"Flow adjustment for '{n}' refers to a flow that is not present in the model."
            assert n in flow_names, msg

        for fadj in strat.flow_adjustments.values():
            for _, source_strata, dest_strata in fadj:
                # Verify source
                self._strata_exist(source_strata)
                # Veryify dest
                self._strata_exist(dest_strata)

        # Validate infectiousness adjustments.
        msg = "All stratification infectiousness adjustments must refer to a compartment that is \
               present in model."
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
            msg = "An infection strain stratification has already been applied, cannot use this \
                   more than once."
            assert not any([s.is_strain() for s in self._stratifications]), msg
            self._disease_strains = strat.strata

        for c in strat.compartments:
            if c not in self._original_compartment_names:
                raise Exception("Trying to stratify non-existent compartment", c)
        # Stratify compartments, split according to split_proportions
        prev_compartment_names = copy.copy(self.compartments)
        self.compartments = strat._stratify_compartments(self.compartments)
        # if not self._defer_actions or self._finalized:
        #    try:
        #        self.initial_population = strat._stratify_compartment_values(
        #            prev_compartment_names, self.initial_population
        #        )
        #    except Exception as e:
        #        logger.critical("Parameterized models must set takes_params in constructor")
        #        raise e

        # Update the cache of compartment names; these need to correct whenever we add a new flow
        self._update_compartment_name_map()

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

            # Only allow age stratification to be applied with complete stratifications,
            # because everyone has an age.
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
        self.stratifications[strat.name] = strat

        self.tracker.append_action(ActionType.STRATIFY, strat=strat)

    def _strata_exist(self, strata: dict):
        """
        Verify whether all the strata exist within the model
        Raises an Exception if not
        """
        strat_names = [s.name for s in self._stratifications]
        for k, v in strata.items():
            if k not in strat_names:
                raise KeyError(f"Invalid stratification {k}")
            for s in self._stratifications:
                if k == s.name:
                    if v not in s.strata:
                        raise ValueError(f"Invalid stratum {v} for {s}")

    def get_stratification(self, name: str) -> Stratification:
        """Return Stratification matching name, or None if not found

        Args:
            name: Name of stratificaton to get

        Returns:
            Stratification
        """
        for stratification in self._stratifications:
            if stratification.name == name:
                return stratification
        return None

    def adjust_population_split(self, strat: str, dest_filter: dict, proportions: dict):
        """Adjust the initial population to redistribute the population for a particular
        stratification, over a subset of some other strata

        Args:
            strat (str): The stratification to redistribute over
            dest_filter (dict): Subset of (other) strata to filter the split by
            proportions (dict): Proportions of new split (must have all strata specified)

        """
        self._assert_not_finalized()

        msg = f"No stratification {strat} found in model"
        assert strat in [s.name for s in self._stratifications], msg

        model_strat = [s for s in self._stratifications if s.name == strat][0]

        msg = "All strata must be specified in proportions"
        assert set(model_strat.strata) == set(proportions), msg

        msg = "Proportions must sum to 1.0"
        np.testing.assert_allclose(sum(proportions.values()), 1.0, err_msg=msg)


        # strat_comps = [c for c in self.compartments if strat in c.strata]
        # # Filter by only the subset we're setting in split_map
        # strat_comps = filter_by_strata(strat_comps, dest_filter)

        # usg = get_unique_strat_groups(strat_comps, strat)

        # for g in usg:
        #     mcomps = self._get_matching_compartments(g.name, g.strata)
        #     idx = [c.idx for c in mcomps]
        #     total = self.initial_population[idx].sum()
        #     for c in mcomps:
        #         k = c.strata[strat]
        #         target_prop = proportions[k]
        #         self.initial_population[c.idx] = total * target_prop

        self.tracker.append_action(
            ActionType.ADJUST_POP_SPLIT,
            strat=strat,
            dest_filter=dest_filter,
            proportions=proportions,
        )

    """
    Running the model
    """
    def _get_step_test(self, parameters: dict=None):
        self._update_compartment_indices()
        self.finalize()

        self._set_backend("jax")
        # self._backend.prepare_to_run(parameters)
        self._backend.prepare_structural()

        from summer2.runner.jax.model_impl import build_run_model

        jax_run_func, jax_runner_dict = build_run_model(
            self._backend, base_params=parameters
        )

        return jax_runner_dict["one_step"](parameters)

    def get_runner(self, parameters: dict, dyn_params: List = None, jit=True, **backend_args):
        self._update_compartment_indices()
        self.finalize()

        self._set_backend("jax")
        # self._backend.prepare_to_run(parameters)
        self._backend.prepare_structural()

        from summer2.runner.jax.model_impl import build_run_model

        jax_run_func, jax_runner_dict = build_run_model(
            self._backend, base_params=parameters, dyn_params=dyn_params, **backend_args
        )
        if jit:
            from jax import jit as jjit

            jax_run_func = jjit(jax_run_func)

        return ModelResults(self, jax_run_func)

    def run(
        self,
        parameters: dict = None,
        solver: str = SolverType.SOLVE_IVP,
        backend_args: dict = None,
        **kwargs,
    ):
        """
        Runs the model over the provided time span, calculating the outputs and the derived outputs.
        The model calculates the outputs by solving an ODE which is defined by the initial
        population and the inter-compartmental flows.

        **Note**: The default ODE solver used to produce the model's outputs does not necessarily
        evaluate every requested timestep. This adaptive
        solver can skip over times, or double back when trying to characterize the ODE. The final
        results are produced by interpolating the
        solution produced by the ODE solver. This means that model dynamics that only occur in
        short time periods may not be reflected in the outputs.

        Args:
            solver (optional): The ODE solver to use, defaults to SciPy's IVP solver.
            **kwargs (optional): Extra arguments to supplied to the solver, see ``summer.solver``
            for details.

        """

        # Ensure we call this before model runs, since it is now disabled inside individual
        # flow constructors
        self._update_compartment_indices()
        self.finalize()

        parameters = parameters or {}

        parameters = {k: v for k, v in parameters.items() if k in self.get_input_parameters()}

        if self._runner is None:
            self._set_backend("jax", backend_args)
            self._backend.prepare_structural()
            self._runner = self.get_runner(parameters, solver=solver, **kwargs)

        self._runner.run(parameters=parameters)

    def _set_backend(self, backend: str, backend_args: dict = None):
        backend_args = backend_args or {}
        self._backend = ModelBackend(self, **backend_args)

    def _solve_ode(self, solver, solver_args: dict):
        """
        Runs the model over the provided time span, calculating the outputs.
        Uses an ODE interpretation of flow rates.
        """

        # Calculate the outputs (compartment sizes) by solving the ODE defined by
        # _get_compartment_rates().
        self.outputs = solve_ode(
            solver,
            self._backend.get_compartment_rates,
            self.initial_population,
            self.times,
            solver_args,
        )

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
        This can be useful when you only care about some results and you want to cut down on
        runtime.
        For example, we may only need some derived outputs for calibration, but may need more later
        when we want to know all the dyanmics that the model actually showed.

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
            model=self,
            whitelist=self._derived_outputs_whitelist,
            baseline=self._baseline,
            idx_cache=self._derived_outputs_idx_cache,
            parameters=self._backend.parameters,
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
            raw_results (optional): Whether to use raw interpolated flow rates, or post-process them
                so that they're more representative of the changes in compartment sizes.
                Defaults to ``False``.
        """
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}

        if self._should_validate:
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

        if isinstance(compartments, str):
            compartments = [compartments]

        if self._should_validate:
            msg = f"A derived output named {name} already exists."
            assert name not in self._derived_output_requests, msg
            is_match_exists = any(
                [
                    any([c.is_match(name, strata) for name in compartments])
                    for c in self.compartments
                ]
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
        Adds a derived output to the model's results. The output will be the aggregate of other
        derived outputs.

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
        Adds a derived output to the model's results. The output will be the accumulated values of
        another derived output over the model's time period.

        Args:
            name: The name of the derived output.
            source: The name of the derived outputs to accumulate.
            start_time (optional): The time to start accumulating from, defaults to model start time
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
            sources: The derived outputs to input into the function.
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

    def request_param_function_output(
        self, name: str, func: params.Function, save_results: bool = True
    ):
        """Request a generic Function output

        Args:
            name (str): _description_
            func (Function): The Function, whose args are Param
            save_results (bool, optional): _description_. Defaults to True.
        """

        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg
        for k, v in func.kwargs.items():
            if isinstance(v, params.DerivedOutput):
                source = v.key
                assert (
                    source in self._derived_output_requests
                ), f"Source {source} has not been requested."
                self._derived_output_graph.add_edge(source, name)

        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.PARAM_FUNCTION,
            "func": func,
            "save_results": save_results,
        }

    def request_computed_value_output(self, name: str, save_results: bool = True):
        """
        Save a computed value process output to derived outputs

        Args:
            name (str): Name (key) of computed value process
            save_results (bool, optional): Save outputs (or discard if False)
        """
        msg = f"A derived output named {name} already exists."
        assert name not in self._derived_output_requests, msg

        self._derived_output_graph.add_node(name)
        self._derived_output_requests[name] = {
            "request_type": DerivedOutputRequest.COMPUTED_VALUE,
            "name": name,
            "save_results": save_results,
        }

    def add_computed_value_process(self, name: str, processor):
        """
        Calculate (at runtime) values derived from the current compartment values and/or
        functions/input data
        providing time varying shared values.  The output values of these processes can be used by
        function parameters, adjustments, and function flows.

        Args:
            name (str): Name (key) of derived value (use this when referencing it in functions)
            processor (DerivedValueProcessor): Object providing implementation
        """
        # FIXME: We might actually have to keep this for now, at least until modellers get sick of
        # seeing it and change over all the code
        raise DeprecationWarning("Use add_computed_value_func instead")

    def add_computed_value_func(self, name: str, func: params.Function):
        if name in self._computed_values_graph_dict:
            raise Exception(f"Computed value function with name {name} already exists")
        self._computed_values_graph_dict[name] = func

    def get_computed_value_keys(self):
        return list(self._computed_values_graph_dict)

    def _get_ref_idx(self):
        if self.ref_date:
            times = ref_times_to_dti(self.ref_date, self.times)
        else:
            times = self.times
        return times

    def get_outputs_df(self):
        idx = self._get_ref_idx()
        column_str = [str(c) for c in self.compartments]
        return pd.DataFrame(self.outputs, index=idx, columns=column_str)

    def get_derived_outputs_df(self):
        idx = self._get_ref_idx()
        return pd.DataFrame(self.derived_outputs, index=idx)

    def get_input_parameters(self):
        if not self._finalized:
            raise Exception("Cannot trace input parameters before model is finalized")
        all_in_var = set(self.graph.get_input_variables())
        all_in_var = all_in_var.union(set(self._do_tracker_graph.get_input_variables()))
        return set([v.key for v in all_in_var if v.source == "parameters"])

    def query_compartments(self, query: dict = None, tags: List = None, as_idx=False):
        from summer2.inspect import query_compartments

        return query_compartments(self, query, tags, as_idx)

    def query_flows(
        self,
        flow_name: str = None,
        source: dict = None,
        dest: dict = None,
        tags: List = None,
    ):
        from summer2.inspect import query_flows

        return query_flows(self, flow_name, source, dest, tags)


class ModelResults:
    def __init__(self, model, run_func):
        self.model = model
        self._run_func = run_func
        self._input_params = model.get_input_parameters()
        self._derived_outputs_idx_cache = None

    def get_outputs_df(self):
        return self.model.get_outputs_df()

    def get_derived_outputs_df(self):
        return self.model.get_derived_outputs_df()

    def run(self, parameters: dict, filter=True):
        if filter:
            parameters = {k: v for k, v in parameters.items() if k in self._input_params}

        results = self._run_func(parameters=parameters)
        
        self.outputs = np.array(results["outputs"])
        self.derived_outputs = {k: np.array(v) for k, v in results["derived_outputs"].items()}
        self.model.outputs = self.outputs
        self.model.derived_outputs = self.derived_outputs
        return results
