"""Implementation of CompartmentalModel and ModelBackend internals in Jax

This is a mess right now!
"""

from functools import partial

from jax import jit, numpy as jnp
from summer2.runner.jax import ode
from summer2.runner.jax import solvers

from summer2.adjust import Overwrite

from summer2.runner import ModelBackend

from summer2.solver import SolverType, SolverArgs

from .stratify import get_calculate_initial_pop
from .derived_outputs import build_derived_outputs_runner


def clean_compartments(compartment_values: jnp.ndarray):
    return jnp.where(compartment_values < 0.0, 0.0, compartment_values)


def get_strain_infection_values(
    strain_infectious_values,
    strain_compartment_infectiousness,
    strain_category_indexer,
    mixing_matrix,
    category_populations,
):
    infected_values = strain_infectious_values * strain_compartment_infectiousness
    infectious_populations = jnp.sum(infected_values[strain_category_indexer], axis=-1)
    infection_density = mixing_matrix @ infectious_populations
    category_prevalence = infectious_populations / category_populations
    infection_frequency = mixing_matrix @ category_prevalence

    return {"infection_density": infection_density, "infection_frequency": infection_frequency}


def build_get_infectious_multipliers(runner):
    population_cat_indexer = jnp.array(runner._population_category_indexer)

    # FIXME: We are hardcoding this for frequency only right now
    infect_proc_type = runner._infection_process_type

    if infect_proc_type == "both":
        raise NotImplementedError(
            "No support for mixed infection frequency/density"
        )

    # FIXME: This could desparately use a tidy-up - all the indexing is a nightmare
    def get_infectious_multipliers(
        time, compartment_values, cur_graph_outputs, compartment_infectiousness
    ):

        infection_frequency = {}
        infection_density = {}

        full_multipliers = jnp.ones(len(runner.infectious_flow_indices))

        mixing_matrix = cur_graph_outputs["mixing_matrix"]
        category_populations = compartment_values[population_cat_indexer].sum(axis=1)

        for strain_idx, strain in enumerate(runner.model._disease_strains):

            strain_compartment_infectiousness = compartment_infectiousness[strain]
            strain_infectious_idx = runner._strain_infectious_indexers[strain]
            strain_category_indexer = runner._strain_category_indexers[strain]

            strain_infectious_values = compartment_values[strain_infectious_idx]
            strain_values = get_strain_infection_values(
                strain_infectious_values,
                strain_compartment_infectiousness,
                strain_category_indexer,
                mixing_matrix,
                category_populations,
            )
            infection_frequency[strain] = strain_values["infection_frequency"]
            infection_density[strain] = strain_values["infection_density"]
            
            if infect_proc_type == "freq":
                strain_ifect = strain_values["infection_frequency"]
            elif infect_proc_type == "dens":
                strain_ifect = strain_values["infection_density"]

            # FIXME: So we produce strain infection values _per category_
            # (ie not all model compartments, not all infectious compartments)
            # We need to rebroadcast these back out the appropriate compartments somehow
            # ... hence the following bit of weird double-dip indexing

            # _infect_strain_lookup_idx is an array of size (num_infectious_comps), whose values
            # are the strain_idx of the strain to whom they belong

            strain_ifectcomp_mask = runner._infect_strain_lookup_idx == strain_idx

            # _infect_cat_lookup_idx is an array of size (num_infectious_comps), whose values
            # are the mixing category to which they belong (and therefore an index into
            # the values returned in strain_ifect above)

            # Get the strain infection values broadcast to num_infectious_comps
            strain_ifect_bcast = strain_ifect[runner._infect_cat_lookup_idx]

            # full_multipliers is the length of the infectious compartments - ie the same as the
            # above mask
            # full_multipliers = full_multipliers.at[strain_ifectcomp_mask].mul(
            #    strain_ifect_bcast[strain_ifectcomp_mask]
            # )

            full_multipliers = full_multipliers.at[strain_ifectcomp_mask].set(
                full_multipliers[strain_ifectcomp_mask] * strain_ifect_bcast[strain_ifectcomp_mask]
            )

        return full_multipliers

    return get_infectious_multipliers


def build_get_flow_weights(runner: ModelBackend):

    m = runner.model

    if "model_variables.time" in m.graph.dag:
        tvkeys = list(m.graph.filter(sources="model_variables.time").dag)
        tv_flow_map = {
            k: m._flow_key_map[k] for k in set(m._flow_key_map).intersection(set(tvkeys))
        }
    else:
        tv_flow_map = {}

    def get_flow_weights(cur_graph_outputs, static_flow_weights):

        flow_weights = jnp.copy(static_flow_weights)

        for k, v in tv_flow_map.items():
            val = cur_graph_outputs[k]
            flow_weights = flow_weights.at[v].set(val)

        return flow_weights

    return get_flow_weights


def _build_calc_computed_values(runner):
    def calc_computed_values(compartment_vals, time, parameters):
        model_variables = {"compartment_values": compartment_vals, "time": time}

        computed_values = runner.computed_values_runner(
            parameters=parameters, model_variables=model_variables
        )

        return computed_values

    return calc_computed_values


def build_get_flow_rates(runner, ts_graph_func):

    # calc_computed_values = build_calc_computed_values(runner)
    get_flow_weights = build_get_flow_weights(runner)
    
    infect_proc_type = runner._infection_process_type
    if infect_proc_type:
        get_infectious_multipliers = build_get_infectious_multipliers(runner)

    # flow_weights_base = jnp.array(runner.flow_weights)

    population_idx = jnp.array(runner.population_idx)
    infectious_flow_indices = jnp.array(runner.infectious_flow_indices)

    def get_flow_rates(compartment_values: jnp.array, time, static_graph_vals, model_data):

        # COULD BE JITTED
        compartment_values = clean_compartments(compartment_values)

        # runner._prepare_time_step(time, compartment_vals)
        sources = {
            "model_variables": {"time": time, "compartment_values": compartment_values},
            "static_inputs": static_graph_vals,
        }

        cur_graph_outputs = ts_graph_func(**sources)

        # JITTED
        # computed_values = calc_computed_values(compartment_values, time, parameters)

        # JITTED
        flow_weights = get_flow_weights(cur_graph_outputs, model_data["static_flow_weights"])

        populations = compartment_values[population_idx]

        # Update for special cases (population-independent and CrudeBirth)
        if runner._has_non_pop_flows:
            populations = populations.at[runner._non_pop_flow_idx].set(1.0)
        if runner._has_crude_birth:
            populations = populations.at[runner._crude_birth_idx].set(compartment_values.sum())

        flow_rates = flow_weights * populations

        # Calculate infection flows
        # JITTED
        if infect_proc_type:
            infect_mul = get_infectious_multipliers(
                time, compartment_values, cur_graph_outputs, model_data["compartment_infectiousness"]
            )
            flow_rates = flow_rates.at[infectious_flow_indices].mul(infect_mul)

                # ReplacementBirthFlow depends on death flows already being calculated; update here
        if runner._has_replacement:
            # Only calculate timestep_deaths if we use replacement, it's expensive...
            _timestep_deaths = flow_rates[runner.death_flow_indices].sum()
            flow_rates = flow_rates.at[runner._replacement_flow_idx].set(_timestep_deaths)


        return flow_rates, cur_graph_outputs["computed_values"]

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


def build_get_rates(runner, ts_graph_func):
    get_flow_rates = build_get_flow_rates(runner, ts_graph_func)
    get_compartment_rates = build_get_compartment_rates(runner)

    def get_rates(compartment_values, time, static_graph_vals, model_data):
        flow_rates, _ = get_flow_rates(compartment_values, time, static_graph_vals, model_data)
        comp_rates = get_compartment_rates(compartment_values, flow_rates)

        return flow_rates, comp_rates

    return {"get_flow_rates": get_flow_rates, "get_rates": get_rates}


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


def build_get_compartment_infectiousness(model):
    """
    Build a Jax function to return the compartment infectiousness (for all compartments),
    of the strain specified by strain
    """

    # This is run during prepare_dynamic
    # i.e. it is done once at the start of a model run, but
    # is parameterized (non-structural)
    def get_compartment_infectiousness(static_graph_values):

        # Find the infectiousness multipliers for each compartment being implemented in the model.
        compartment_infectiousness = jnp.ones(len(model.compartments))

        # Apply infectiousness adjustments
        for strat in model._stratifications:
            for comp_name, adjustments in strat.infectiousness_adjustments.items():
                for stratum, adjustment in adjustments.items():
                    if adjustment:
                        is_overwrite = isinstance(adjustment, Overwrite)
                        adj_value = static_graph_values[adjustment.param._graph_key]
                        adj_comps = model.get_matching_compartments(
                            comp_name, {strat.name: stratum}
                        )
                        for c in adj_comps:
                            if is_overwrite:
                                compartment_infectiousness = compartment_infectiousness.at[
                                    c.idx
                                ].set(adj_value)
                            else:
                                orig_value = compartment_infectiousness[c.idx]
                                compartment_infectiousness = compartment_infectiousness.at[
                                    c.idx
                                ].set(adj_value * orig_value)

        strain_comp_inf = {}

        for strain in model._disease_strains:
            if "strain" in model.stratifications:
                strain_filter = {"strain": strain}
            else:
                strain_filter = {}

            # _Must_ be ordered here
            strain_infect_comps = model.query_compartments(
                strain_filter, tags="infectious", as_idx=True
            )

            strain_comp_inf[strain] = compartment_infectiousness[strain_infect_comps]

        return strain_comp_inf

    return get_compartment_infectiousness


def build_run_model(runner, base_params=None, dyn_params=None, solver=None, solver_args=None):

    if dyn_params is None:
        dyn_params = runner.model.graph.get_input_variables()
    else:
        dyn_params = [
            f"parameters.{p}" if not p.startswith("parameters.") else p for p in dyn_params
        ]

    # Graph frozen for all non-calibration parameters
    if base_params is None:
        base_params = {}

    source_inputs = {"parameters": base_params}

    ts_vars = runner.model.graph.query("model_variables")

    dyn_params = set(dyn_params).union(set(ts_vars))

    param_frozen_cg = runner.model.graph.freeze(dyn_params, source_inputs)

    # static_cg = param_frozen_cg.filter(exclude=ts_vars)
    # static_graph_func = static_cg.get_callable()(parameters=base_params)

    timestep_cg, static_cg = param_frozen_cg.freeze(ts_vars)

    timestep_graph_func = jit(timestep_cg.get_callable())
    static_graph_func = static_cg.get_callable()

    rates_funcs = build_get_rates(runner, timestep_graph_func)
    get_rates = rates_funcs["get_rates"]
    get_flow_rates = rates_funcs["get_flow_rates"]

    from jax import vmap

    get_flows_for_outputs = vmap(get_flow_rates, in_axes=(0, 0, None, None), out_axes=(0))

    def get_comp_rates(comp_vals, t, static_graph_vals, model_data):
        return get_rates(comp_vals, t, static_graph_vals, model_data)[1]

    if solver is None or solver == SolverType.SOLVE_IVP:
        solver = SolverType.ODE_INT

    if solver == SolverType.ODE_INT:
        if solver_args is None:
            # Some sensible defaults; faster than
            # the odeint defaults,
            # but accurate enough for our tests
            solver_args = SolverArgs.DEFAULT


        def get_ode_solution(initial_population, times, static_graph_vals, model_data):
            return ode.odeint(
                get_comp_rates,
                initial_population,
                times,
                static_graph_vals,
                model_data,
                **solver_args
            )

    elif solver == SolverType.RUNGE_KUTTA:

        def get_ode_solution(initial_population, times, static_graph_vals, model_data):
            return solvers.rk4(
                get_comp_rates, initial_population, times, static_graph_vals, model_data
            )

    elif solver == SolverType.EULER:

        def get_ode_solution(initial_population, times, static_graph_vals, model_data):
            return solvers.euler(
                get_comp_rates, initial_population, times, static_graph_vals, model_data
            )

    else:
        raise NotImplementedError("Incompatible SolverType for Jax runner", solver)

    times = jnp.array(runner.model.times)

    calc_initial_pop = get_calculate_initial_pop(runner.model)
    get_compartment_infectiousness = build_get_compartment_infectiousness(runner.model)

    calc_derived_outputs = build_derived_outputs_runner(runner.model)

    m = runner.model
    if "model_variables.time" in m.graph.dag:
        tv_keys = list(m.graph.filter(sources="model_variables.time").dag)
    else:
        tv_keys = []

    static_flow_map = {k: m._flow_key_map[k] for k in set(m._flow_key_map).difference(set(tv_keys))}

    def run_model(parameters):

        static_graph_vals = static_graph_func(parameters=parameters)
        initial_population = calc_initial_pop(static_graph_vals)

        static_flow_weights = jnp.zeros(len(runner.model._flows))
        for k, v in static_flow_map.items():
            val = static_graph_vals[k]
            static_flow_weights = static_flow_weights.at[v].set(val)

        compartment_infectiousness = get_compartment_infectiousness(static_graph_vals)
        model_data = {
            "compartment_infectiousness": compartment_infectiousness,
            "static_flow_weights": static_flow_weights,
        }

        outputs = get_ode_solution(initial_population, times, static_graph_vals, model_data)

        out_flows, out_cv = get_flows_for_outputs(outputs, times, static_graph_vals, model_data)

        model_variables = {"outputs": outputs, "flows": out_flows, "computed_values": out_cv}

        derived_outputs = calc_derived_outputs(
            parameters=parameters, model_variables=model_variables
        )

        # return {"outputs": outputs, "model_data": model_data}
        return {
            "outputs": outputs,
            "derived_outputs": derived_outputs,
        }  # "model_data": model_data}

    def one_step(parameters: dict=None):

        static_graph_vals = static_graph_func(parameters=parameters)
        initial_population = calc_initial_pop(static_graph_vals)

        static_flow_weights = jnp.zeros(len(runner.model._flows))
        for k, v in static_flow_map.items():
            val = static_graph_vals[k]
            static_flow_weights = static_flow_weights.at[v].set(val)

        compartment_infectiousness = get_compartment_infectiousness(static_graph_vals)
        model_data = {
            "compartment_infectiousness": compartment_infectiousness,
            "static_flow_weights": static_flow_weights,
        }

        flow_rates, comp_rates = get_rates(initial_population, runner.model.times[0], static_graph_vals, model_data)

        # return {"outputs": outputs, "model_data": model_data}
        return {
            "flow_rates": flow_rates,
            "comp_rates": comp_rates,
            "initial_population": initial_population
        }  # "model_data": model_data}

    runner_dict = {
        "get_rates": get_rates,
        "get_flow_rates": get_flow_rates,
        "get_comp_rates": get_comp_rates,
        "calc_initial_pop": calc_initial_pop,
        "get_compartment_infectiousness": get_compartment_infectiousness,
        "get_ode_solution": get_ode_solution,
        "calc_derived_outputs": calc_derived_outputs,
        "timestep_graph_func": timestep_graph_func,
        "timestep_cg": timestep_cg,
        "static_cg": static_cg,
        "static_graph_func": static_graph_func,
        "one_step": one_step
    }

    return run_model, runner_dict
