import numpy as np

import summer.flows as flows
from summer.runner.model_runner import ModelRunner


class JaxRunner(ModelRunner):
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

        self._precompute_flow_maps()
        self._precompute_flow_weights()

        self._build_infectious_multipliers_lookup()

    def prepare_dynamic(self, parameters: dict = None):
        """This is just here to appease the ABC
        All the calculation that would occur here happens in the returned runner function
        """

    def _precompute_flow_weights(self):
        """Calculate all static flow weights before running,
        and build indices for time-varying weights"""
        self.flow_weights = np.zeros(len(self.model._flows))
        time_varying_weight_indices = []
        for i, f in self._iter_non_function_flows:
            # FIXME:
            # Unlike vectorized runner, we just blindly calculate
            # everything every timestep.  It's a bad scene,
            # but let's just get things working for now...

            # weight_type = f.weight_type()
            # if weight_type == flows.WeightType.STATIC:
            #    weight = f.get_weight_value(0, None, self.parameters)
            #    self.flow_weights[i] = weight
            # elif weight_type == flows.WeightType.FUNCTION:
            time_varying_weight_indices.append(i)

        self.time_varying_weight_indices = np.array(time_varying_weight_indices, dtype=int)

        self._map_blocks()

    def _map_blocks(self):
        flow_block_maps = {}
        for i in self.time_varying_weight_indices:
            f = self.model._flows[i]

            def get_key(f):
                if isinstance(f.param, list):
                    param = tuple(f.param)
                else:
                    param = f.param
                return (param, tuple(f.adjustments))

            key = get_key(f)
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

    def get_compartment_rates(self, compartment_values: np.ndarray, time: float) -> np.ndarray:
        # Solely implemented to appease the ABC requirements
        pass

    def get_flow_rates(self, compartment_values: np.ndarray, time: float) -> np.ndarray:
        # Solely implemented to appease the ABC requirements
        pass
