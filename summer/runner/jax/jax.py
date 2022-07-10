import os

from jax import numpy as jnp
from jax.config import config as jax_config

from summer.parameters import get_model_param_value

# Jax configuration

jax_config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# What already works as JIT
# _get_mixing_matrix
# _calc_computed_values
#
# What is a mess
# Calculate strain infection values


def clean_compartments(compartment_values: jnp.ndarray):
    return jnp.where(compartment_values < 0.0, 0.0, compartment_values)


def build_get_mixing_matrix(runner):
    def get_mixing_matrix(time: float, parameters) -> jnp.ndarray:
        """
        Returns the final mixing matrix for a given time.
        """
        # We actually have some matrices, let's do things with them...
        if len(runner.model._mixing_matrices):
            mixing_matrix = None
            for mm_func in runner.model._mixing_matrices:
                # Assume each mixing matrix is either an np.ndarray or
                # a function of time that returns one.
                # mm = mm_func(time) if callable(mm_func) else mm_func
                mm = get_model_param_value(mm_func, time, None, parameters)
                # Get Kronecker product of old and new mixing matrices.
                # Only do this if we actually need to
                if mixing_matrix is None:
                    mixing_matrix = mm
                else:
                    mixing_matrix = jnp.kron(mixing_matrix, mm)
        else:
            mixing_matrix = runner.model._DEFAULT_MIXING_MATRIX

        return mixing_matrix

    return get_mixing_matrix


def get_strain_infection_values(
    compartment_values,
    strain_compartment_infectiousness,
    category_matrix,
    mixing_matrix,
    category_populations,
):
    infected_values = compartment_values * strain_compartment_infectiousness
    infectious_populations = category_matrix @ infected_values
    infection_density = mixing_matrix @ infectious_populations
    category_prevalence = infectious_populations / category_populations
    infection_frequency = mixing_matrix @ category_prevalence

    return infection_density, infection_frequency


def build_get_infectious_multipliers(runner):
    SCI = runner._compartment_infectiousness["default"]
    cat_mat = runner._category_matrix
    get_mixing_matrix = build_get_mixing_matrix(runner)

    def get_infectious_multipliers(time, compartment_values, parameters):
        mm = get_mixing_matrix(time, parameters)
        cat_pops = cat_mat @ compartment_values
        return get_strain_infection_values(compartment_values, SCI, cat_mat, mm, cat_pops)[1]

    return get_infectious_multipliers


def build_get_flow_weights(runner):

    flow_block_maps = flatten_fbm(runner.flow_block_maps)

    def get_flow_weights(static_flow_weights, computed_values, parameters, time):
        flow_weights = static_flow_weights.copy()
        for (param, adjustments, flow_idx) in flow_block_maps:
            value = get_model_param_value(param, time, computed_values, parameters)
            for a in adjustments:
                value = a.get_new_value(value, computed_values, time, parameters)
            flow_weights = flow_weights.at[flow_idx].set(value)
        return flow_weights

    return get_flow_weights


def flatten_fbm(flow_block_map):
    out_map = []
    for (param, adjustments), flow_idx in flow_block_map.items():
        out_map.append((param, adjustments, flow_idx))
    return out_map


def build_calc_computed_values(runner):
    def calc_computed_values(compartment_vals, time, parameters):
        model_variables = {"compartment_values": compartment_vals, "time": time}

        computed_values = runner.computed_values_runner(
            parameters=parameters, model_variables=model_variables
        )

        return computed_values

    return calc_computed_values


def build_get_flow_rates(runner):

    calc_computed_values = build_calc_computed_values(runner)
    get_flow_weights = build_get_flow_weights(runner)
    get_infectious_multipliers = build_get_infectious_multipliers(runner)

    flow_weights_base = jnp.array(runner.flow_weights)

    population_idx = jnp.array(runner.population_idx)
    infectious_flow_indices = jnp.array(runner.infectious_flow_indices)

    def get_flow_rates(compartment_values: jnp.array, time, parameters):

        # COULD BE JITTED
        compartment_values = clean_compartments(compartment_values)

        # runner._prepare_time_step(time, compartment_vals)

        # JITTED
        computed_values = calc_computed_values(compartment_values, time, parameters)

        # JITTED
        flow_weights = get_flow_weights(flow_weights_base, computed_values, parameters, time)

        populations = compartment_values[population_idx]

        # Update for special cases (population-independent and CrudeBirth)
        if runner._has_non_pop_flows:
            populations = populations.at[runner._non_pop_flow_idx].set(1.0)
        if runner._has_crude_birth:
            populations = populations.at[runner._crude_birth_idx].set(compartment_values.sum())

        flow_rates = flow_weights * populations

        # Calculate infection flows
        # JITTED
        infect_mul = get_infectious_multipliers(time, compartment_values, parameters)
        flow_rates = flow_rates.at[infectious_flow_indices].set(
            flow_rates[infectious_flow_indices] * infect_mul
        )

        return flow_rates

    return get_flow_rates


def build_get_compartment_rates(runner):
    accum_maps = get_accumulation_maps(runner)

    def get_compartment_rates(compartment_values, flow_rates):
        comp_rates = jnp.zeros_like(compartment_values)

        for (flow_src, comp_target) in accum_maps["positive"]:
            comp_rates = comp_rates.at[comp_target].add(flow_rates[flow_src])
        for (flow_src, comp_target) in accum_maps["negative"]:
            comp_rates = comp_rates.at[comp_target].add(-flow_rates[flow_src])

        return comp_rates

    return get_compartment_rates


def build_get_rates(runner):
    get_flow_rates = build_get_flow_rates(runner)
    get_compartment_rates = build_get_compartment_rates(runner)

    def get_rates(compartment_values, time, parameters):
        flow_rates = get_flow_rates(compartment_values, time, parameters)
        comp_rates = get_compartment_rates(compartment_values, flow_rates)

        return comp_rates, flow_rates

    return get_rates


def get_accumulation_maps(runner):
    pos_map = [mflow for mflow in runner._pos_flow_map]
    neg_map = [mflow for mflow in runner._neg_flow_map]

    def peel_flow_map(flow_map):
        targets = []
        src_idx = []
        remainder = []
        for (src_flow, target) in flow_map:
            if target not in targets:
                targets.append(target)
                src_idx.append(src_flow)
            else:
                remainder.append((src_flow, target))
        return jnp.array(src_idx), jnp.array(targets), remainder

    def recurse_unpeel(flow_map):
        remainder = flow_map
        full_map = []
        while len(remainder) > 0:
            sources, targets, remainder = peel_flow_map(remainder)
            full_map.append((sources, targets))
        return full_map

    return {"positive": recurse_unpeel(pos_map), "negative": recurse_unpeel(neg_map)}
