"""
This module presents classes which are used to define stratifications,
which can be applied to the model.
"""
from typing import Callable, Dict, List, Optional, Union

from numbers import Real

import numpy as np
from computegraph.utils import is_var
from computegraph.types import GraphObject

from summer2.adjust import Multiply, Overwrite, enforce_multiply
from summer2.compartment import Compartment
from summer2.parameters import is_func, get_static_param_value
from summer2.parameters.param_impl import get_modelparameter_from_param

Adjustment = Union[Multiply, Overwrite]
MixingMatrix = Union[np.ndarray, Callable[[float], np.ndarray]]

# Allowable error in requested compartment splits between one and the total requested
COMP_SPLIT_REQUEST_ERROR = 1e-2


class Stratification:
    """
    A generic stratification applied to the compartmental model.

    Args:
        name: The name of the stratification.
        strata: The new strata that will be created.
        compartments: The compartments which will be stratified.

    """

    _is_ageing = False
    _is_strain = False

    def __init__(
        self,
        name: str,
        strata: List[str],
        compartments: List[str],
    ):
        self.name = name
        self.strata = list(map(str, strata))
        self.compartments = [Compartment(c) if type(c) is str else c for c in compartments]

        # Split population evenly by default.
        num_strata = len(self.strata)
        self.population_split = {s: 1 / num_strata for s in self.strata}

        # Flows are not adjusted by default.
        self.flow_adjustments = {}
        self.infectiousness_adjustments = {}

        # Store internal fast lookup table for flow adjustments using frozenset
        self._flow_adjustments_fs = {}

        # No heterogeneous mixing matrix by default.
        self.mixing_matrix = None

        # Enable/disable validation (runtime assertions)
        # Not directly user accessable - is toggled by CompartmentalModel
        self._validate = True

    def __repr__(self):
        return f"Stratification: {self.name}"

    def is_ageing(self) -> bool:
        """Returns ``True`` if this stratification represents a set of age groups
        with ageing dynamics"""
        return self._is_ageing

    def is_strain(self) -> bool:
        """Returns ``True`` if this stratification represents a set of disease strains"""
        return self._is_strain

    def set_population_split(self, proportions: Dict[str, float]):
        """
        Sets how the stratification will split the population between the strata.

        Args:
            proportions: A map of proportions to be assigned to each strata.

        The supplied proportions must:

            - Specify all strata
            - All be positive
            - Sum to 1 +/- error defined above

        """
        if all([isinstance(v, Real) for v in proportions.values()]):
            self.validate_population_split(proportions)

        self.population_split = proportions

    def validate_population_split(self, proportions: dict):
        msg = f"All strata must be specified when setting population split: {proportions}"
        assert set(list(proportions.keys())) == set(self.strata), msg
        msg = f"All proportions must be >= 0 when setting population split: {proportions}"
        assert all([v >= 0 for v in proportions.values()]), msg
        msg = f"All proportions sum to 1+/-{COMP_SPLIT_REQUEST_ERROR} when setting \
            population split: {proportions}"
        assert abs(1 - sum(proportions.values())) < COMP_SPLIT_REQUEST_ERROR, msg

    def set_flow_adjustments(
        self,
        flow_name: str,
        adjustments: Dict[str, Adjustment],
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
    ):
        """
        Set an adjustment of a flow to the stratification.
        Adjustments from previous stratifications will be applied before this.
        You can use time-varying functions for infectiousness adjustments.

        It is possible to specify multiple flow adjustments for the same flow in a
        given Stratification.
        In this case, only the last-created applicable adjustment will be applied.

        Args:
            flow_name: The name of the flow to adjust.
            adjustments: A dict of adjustments to apply to the flow.
            source_strata (optional): A whitelist of strata to filter the target flow's
            source compartments.
            dest_strata (optional): A whitelist of strata to filter the target flow's
            destination compartments.

        Example:
            Create an adjustment for the 'recovery' flow based on location::

                strat = Stratification(
                    name="location",
                    strata=["urban", "rural", "alpine"],
                    compartments=["S", "I", "R"]
                )
                strat.set_flow_adjustments("recovery", {
                    "urban": Multiply(1.5),
                    "rural": Multiply(0.8),
                    "alpine": None, # No adjustment
                })

        """
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}
        msg = "You must specify all strata when adding flow adjustments."
        assert set(adjustments.keys()) == set(self.strata), msg

        adjustments = {k: enforce_multiply(v) for k, v in adjustments.items()}

        msg = "All flow adjustments must be Multiply, Overwrite or None."
        assert all(
            [
                type(adj) is Overwrite or type(adj) is Multiply or adj is None
                for adj in adjustments.values()
            ]
        ), msg

        msg = "Cannot add new flow adjustments after stratification has already been applied"
        assert len(self._flow_adjustments_fs) == 0, msg

        if flow_name not in self.flow_adjustments:
            self.flow_adjustments[flow_name] = []

        # FIXME Turn this into class or named tuple or something... we want to know what's going on
        self.flow_adjustments[flow_name].append((adjustments, source_strata, dest_strata))

    def get_flow_adjustment(self, flow) -> dict:
        """
        Note that the loop structure implies that if the user has requested multiple adjustments
        that apply to a single combination of strata (across multiple stratifications), then only
        the last one that is applicable will be used - because the last request will over-write the
        earlier ones in the loop.
        Therefore, the most recently added flow adjustment that matches a given flow will be
        returned.

        """
        flow_adjustments = self.flow_adjustments.get(flow.name, [])
        matching_adjustment = None

        # Cache frozenset lookup
        if flow.name not in self._flow_adjustments_fs:
            self._flow_adjustments_fs[flow.name] = cur_fadj_fs = []
            for adjustment, source_strata, dest_strata in flow_adjustments:
                cur_fadj_fs.append(
                    (
                        adjustment,
                        source_strata,
                        frozenset(source_strata.items()),
                        dest_strata,
                        frozenset(dest_strata.items()),
                    )
                )

        flow_adj_fs = self._flow_adjustments_fs[flow.name]

        # Loop over all the requested adjustments.
        for adjustment, source_strata, _source_strata, dest_strata, _dest_strata in flow_adj_fs:

            # These are expensive assertions and can be disabled in multi-run situations
            if self._validate:
                # For entry flows:
                msg = f"Source strata requested in flow adjustment of {self.name}, but {flow.name} \
                    does not have a source"
                assert not (source_strata and not flow.source), msg

                # For exit flows:
                msg = f"Dest strata requested in flow adjustment of {self.name}, but {flow.name} \
                    does not have a dest"
                assert not (dest_strata and not flow.dest), msg

            # Make sure that the source request applies to this flow because it has all of the
            # requested strata.
            # Note that these can be specified in the current or any previous stratifications.
            is_source_no_match = (
                source_strata and flow.source and not flow.source._has_strata(_source_strata)
            )
            if is_source_no_match:
                continue
            is_dest_no_match = dest_strata and flow.dest and not flow.dest._has_strata(_dest_strata)
            if is_dest_no_match:
                continue

            matching_adjustment = adjustment

        return matching_adjustment

    def add_infectiousness_adjustments(
        self, compartment_name: str, adjustments: Dict[str, Adjustment]
    ):
        """
        Add an adjustment of a compartment's infectiousness to the stratification.
        You cannot currently use time-varying functions for infectiousness adjustments.
        All strata in this stratification must be specified as keys in the adjustments argument,
        if no adjustment required for a stratum, specify None as the value to the request for
        that stratum.

        Args:
            compartment_name: The name of the compartment to adjust.
            adjustments: An dict of adjustments to apply to the compartment.

        Example:
            Create an adjustment for the 'I' compartment based on location::

                strat = Stratification(
                    name="location",
                    strata=["urban", "rural", "alpine"],
                    compartments=["S", "I", "R"]
                )
                strat.add_infectiousness_adjustments("I", {
                    "urban": adjust.Multiply(1.5),
                    "rural": adjust.Multiply(0.8),
                    "alpine": None, # No adjustment
                })

        """
        msg = "You must specify all strata when adding infectiousness adjustments."
        assert set(adjustments.keys()) == set(self.strata), msg

        adjustments = {k: enforce_multiply(v) for k, v in adjustments.items()}

        # for k, v in adjustments.items():
        #    if v is not None:
        #        v.param = get_modelparameter_from_param(v.param)

        msg = "All infectiousness adjustments must be Multiply, Overwrite or None."
        assert all(
            [type(a) is Overwrite or type(a) is Multiply or a is None for a in adjustments.values()]
        ), msg

        msg = f"An infectiousness adjustment for {compartment_name} \
                already exists for strat {self.name}"
        assert compartment_name not in self.infectiousness_adjustments, msg

        # FIXME: This should work for computegraph Functions that are not time dependant,
        # (and fail for those that are), but it's tricky to check...
        # msg = "Cannot use time varying functions for infectiousness adjustments."
        # assert not any([adj.param.is_time_varying() for adj in adjustments.values() if adj]), msg

        self.infectiousness_adjustments[compartment_name] = adjustments

    def set_mixing_matrix(self, mixing_matrix: MixingMatrix):
        """
        Sets the mixing matrix for the model.
        Note that this must apply to all compartments, although this is checked at runtime rather
        than here.
        """
        msg = "Strain stratifications cannot have a mixing matrix."
        assert not self.is_strain(), msg

        # msg = "Mixing matrix must be a NumPy array, or return a NumPy array."
        # assert (
        #    (isinstance(mixing_matrix, np.ndarray))
        #    or isinstance(mixing_matrix, GraphObject)
        #    or is_func(mixing_matrix)
        # ), msg

        # if isinstance(mixing_matrix, np.ndarray):
        #    num_strata = len(self.strata)
        #    msg = f"Mixing matrix must have both {num_strata} rows and {num_strata} columns."
        #    assert mixing_matrix.shape == (num_strata, num_strata), msg

        self.mixing_matrix = mixing_matrix

    def _stratify_compartments(self, comps: List[Compartment]) -> List[Compartment]:
        """
        Stratify the model compartments into sub-compartments, based on the strata names,
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartments.
        """
        strat_base_indices = []
        passthrough_base_indices = []
        passthrough_target_indices = []
        stratum_target_indices = {s: [] for s in self.strata}

        new_comps = []
        idx = 0
        for base_idx, old_comp in enumerate(comps):
            should_stratify = old_comp.has_name_in_list(self.compartments)
            if should_stratify:
                strat_base_indices.append(base_idx)
                for stratum in self.strata:
                    new_comp = old_comp.stratify(self.name, stratum)
                    new_comp.idx = idx
                    new_comps.append(new_comp)
                    stratum_target_indices[stratum].append(idx)
                    idx += 1
            else:
                passthrough_base_indices.append(base_idx)
                passthrough_target_indices.append(idx)
                old_comp.idx = idx
                new_comps.append(old_comp)
                idx += 1

        self._strat_base_indices = np.array(strat_base_indices, dtype=int)
        self._passthrough_base_indices = np.array(passthrough_base_indices, dtype=int)
        self._passthrough_target_indices = np.array(passthrough_target_indices, dtype=int)
        self._stratum_target_indices = {
            k: np.array(v, dtype=int) for k, v in stratum_target_indices.items()
        }
        self._new_size = idx

        return new_comps

    def _stratify_compartment_values(
        self, comps: List[Compartment], comp_values: np.ndarray, static_graph_values: dict = None
    ) -> np.ndarray:
        """
        Stratify the model compartments into sub-compartments, based on the strata names provided.
        Split the population according to the provided proportions.
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartment values.
        """
        assert len(comps) == len(comp_values)
        new_comp_values = []

        if static_graph_values:
            population_split = get_static_param_value(self.population_split, static_graph_values)
        else:
            population_split = self.population_split

        for idx in range(len(comp_values)):
            should_stratify = comps[idx].has_name_in_list(self.compartments)
            if should_stratify:
                for stratum in self.strata:
                    new_value = comp_values[idx] * population_split[stratum]
                    new_comp_values.append(new_value)
            else:
                new_comp_values.append(comp_values[idx])

        return np.array(new_comp_values)


class AgeStratification(Stratification):
    """
    A stratification that represents a set of age groups with ageing dynamics.

    Args:
        name: The name of the stratification.
        strata: The new strata that will be created.
        compartments: The compartments which will be stratified.


    Strata must be a list of strings or integers that represent all ages. For example, the strata
    [0, 10, 20, 30] will be interpreted as the age groups 0-9, 10-19, 20-29, 30+ respectively.

    Using this stratification will automatically add a set of ageing flows to the model,
    where the exit rate is proportional to the age group's time span. For example, in the 0-9
    strata, there will be a flow created where 10% of occupants "age up" into the 10-19 strata
    each year.
    **Critically**, this feature assumes that each of the model's time steps represents a year.

    """

    _is_ageing = True

    def __init__(
        self,
        name: str,
        strata: List[str],
        compartments: List[str],
    ):
        try:
            _strata = sorted(map(int, strata))
        except Exception:
            raise AssertionError("Strata must be in an int-compatible format")

        assert _strata[0] == 0, "First age strata must be 0"

        _strata = map(str, _strata)
        super().__init__(name, _strata, compartments)


class StrainStratification(Stratification):
    """
    A stratification that represents a set of disease strains.

    Args:
        name: The name of the stratification.
        strata: The new strata that will be created.
        compartments: The compartments which will be stratified.


    Each requested stratum will be interpreted as a different strain of the disease being modelled.
    This will mean that the force of infection calculations will consider each strain separately.

    Strain stratifications cannot use a mixing matrix

    """

    _is_strain = True
