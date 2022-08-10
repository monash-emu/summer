"""Implementation of CompartmentalModel and ModelRunner internals in Jax

This is a mess right now!
"""

from jax import numpy as jnp
from summer.runner.jax import ode
from summer.runner.jax import solvers

from summer.solver import SolverType

from .stratify import get_calculate_initial_pop
from .derived_outputs import build_derived_outputs_runner

from summer.parameters import get_model_param_value
from summer.experimental.abstract_parameter import evaluate_lazy


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
                # mm = mm_func.get_value(time, {}, parameters)
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

    return {"infection_density": infection_density, "infection_frequency": infection_frequency}


def build_get_infectious_multipliers(runner):
    category_matrix = runner._category_matrix
    get_mixing_matrix = build_get_mixing_matrix(runner)

    # FIXME: We are hardcoding this for frequency only right now
    if not runner._infection_frequency_only:
        raise NotImplementedError(
            "Model must have at least one infection frequency flow, and no infection density"
        )

    # FIXME: This could desparately use a tidy-up - all the indexing is a nightmare
    def get_infectious_multipliers(
        time, compartment_values, parameters, compartment_infectiousness
    ):

        infection_frequency = {}
        infection_density = {}

        full_multipliers = jnp.ones(len(runner.infectious_flow_indices))

        mm = get_mixing_matrix(time, parameters)
        cat_pops = category_matrix @ compartment_values

        for strain_idx, strain in enumerate(runner.model._disease_strains):
            strain_infectiousness = compartment_infectiousness[strain]
            strain_values = get_strain_infection_values(
                compartment_values, strain_infectiousness, category_matrix, mm, cat_pops
            )
            infection_frequency[strain] = strain_values["infection_frequency"]
            infection_density[strain] = strain_values["infection_density"]

            strain_ifect = strain_values["infection_frequency"]

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


def build_get_flow_weights(runner):

    flow_block_maps = flatten_fbm(runner.flow_block_maps)

    def get_flow_weights(static_flow_weights, computed_values, parameters, time):
        flow_weights = static_flow_weights.copy()
        for (param, adjustments, flow_idx) in flow_block_maps:
            value = param.get_value(time, computed_values, parameters)

            # value = get_model_param_value(param, time, computed_values, parameters, True)
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

    def get_flow_rates(compartment_values: jnp.array, time, parameters, model_data):

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
        infect_mul = get_infectious_multipliers(
            time, compartment_values, parameters, model_data["compartment_infectiousness"]
        )
        flow_rates = flow_rates.at[infectious_flow_indices].set(
            flow_rates[infectious_flow_indices] * infect_mul
        )

        return flow_rates, computed_values

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

    def get_rates(compartment_values, time, parameters, model_data):
        flow_rates, _ = get_flow_rates(compartment_values, time, parameters, model_data)
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


def build_get_compartment_infectiousness_for_strain(model, strain: str):
    """
    Build a Jax function to return the compartment infectiousness (for all compartments),
    of the strain specified by strain
    """
    # Figure out which compartments should be infectious

    infectious_mask = jnp.array(
        [c.has_name_in_list(model._infectious_compartments) for c in model.compartments]
    )

    # This is run during prepare_dynamic
    # i.e. it is done once at the start of a model run, but
    # is parameterized (non-structural)
    def get_compartment_infectiousness_for_strain(parameters):
        # Find the infectiousness multipliers for each compartment being implemented in the model.
        # Start from assumption that each compartment is not infectious.
        compartment_infectiousness = jnp.zeros(len(model.compartments))
        # Set all infectious compartments to be equally infectious.
        compartment_infectiousness = compartment_infectiousness.at[infectious_mask].set(1.0)

        # Apply infectiousness adjustments.
        for idx, comp in enumerate(model.compartments):
            inf_value = compartment_infectiousness[idx]
            for strat in model._stratifications:
                for comp_name, adjustments in strat.infectiousness_adjustments.items():
                    if comp_name == comp.name:
                        for stratum, adjustment in adjustments.items():
                            should_apply_adjustment = adjustment and comp.has_stratum(
                                strat.name, stratum
                            )
                            if should_apply_adjustment:
                                # Cannot use time-varying functions for infectiousness adjustments,
                                # because this is calculated before the model starts running.
                                inf_value = adjustment.get_new_value(
                                    inf_value, None, None, parameters
                                )

            compartment_infectiousness = compartment_infectiousness.at[idx].set(inf_value)

        if strain != model._DEFAULT_DISEASE_STRAIN:
            # FIXME: If there are multiple strains, but one of them is _DEFAULT_DISEASE_STRAIN
            # there will almost certainly be incorrect masks applied
            # Filter out all values that are not in the given strain.
            strain_mask = jnp.zeros(len(model.compartments))
            for idx, compartment in enumerate(model.compartments):
                if compartment.has_stratum("strain", strain):
                    strain_mask = strain_mask.at[idx].set(1.0)

            compartment_infectiousness = compartment_infectiousness * strain_mask

        return compartment_infectiousness

    return get_compartment_infectiousness_for_strain


def build_compartment_infectiousness_calc(model):
    """Build a functioning return the compartment infectiousness for _all_ strains in the model"""
    strain_funcs = {}
    for strain in model._disease_strains:
        strain_funcs[strain] = build_get_compartment_infectiousness_for_strain(model, strain)

    def get_compartment_infectiousness(parameters):
        compartment_infectiousness = {}
        for strain in model._disease_strains:
            compartment_infectiousness[strain] = strain_funcs[strain](parameters)
        return compartment_infectiousness

    return get_compartment_infectiousness


def build_run_model(runner, solver=None):
    rates_funcs = build_get_rates(runner)
    get_rates = rates_funcs["get_rates"]
    get_flow_rates = rates_funcs["get_flow_rates"]

    from jax import vmap

    get_flows_for_outputs = vmap(get_flow_rates, in_axes=(0, 0, None, None), out_axes=(0))

    def get_comp_rates(comp_vals, t, parameters, model_data):
        return get_rates(comp_vals, t, parameters, model_data)[1]

    if solver is None or solver == SolverType.SOLVE_IVP:
        solver = SolverType.ODE_INT

    if solver == SolverType.ODE_INT:

        def get_ode_solution(initial_population, times, parameters, model_data):
            return ode.odeint(get_comp_rates, initial_population, times, parameters, model_data)

    elif solver == SolverType.RUNGE_KUTTA:

        def get_ode_solution(initial_population, times, parameters, model_data):
            return solvers.rk4(get_comp_rates, initial_population, times, parameters, model_data)

    elif solver == SolverType.EULER:

        def get_ode_solution(initial_population, times, parameters, model_data):
            return solvers.euler(get_comp_rates, initial_population, times, parameters, model_data)

    else:
        raise NotImplementedError("Incompatible SolverType for Jax runner", solver)

    times = jnp.array(runner.model.times)

    calc_initial_pop = get_calculate_initial_pop(runner.model)
    get_compartment_infectiousness = build_compartment_infectiousness_calc(runner.model)

    calc_derived_outputs = build_derived_outputs_runner(runner.model)

    def run_model(parameters):
        lazy_parameters = {
            f"_lazy_{hash(p)}": evaluate_lazy(p, parameters) for p in runner.model._lazy_params
        }

        parameters.update(lazy_parameters)

        initial_population = calc_initial_pop(parameters)
        compartment_infectiousness = get_compartment_infectiousness(parameters)
        model_data = {"compartment_infectiousness": compartment_infectiousness}

        outputs = get_ode_solution(initial_population, times, parameters, model_data)

        out_flows, out_cv = get_flows_for_outputs(outputs, times, parameters, model_data)

        model_variables = {"outputs": outputs, "flows": out_flows, "computed_values": out_cv}

        derived_outputs = calc_derived_outputs(
            parameters=parameters, model_variables=model_variables
        )
        # return {"outputs": outputs, "model_data": model_data}
        return {"outputs": outputs, "derived_outputs": derived_outputs, "model_data": model_data}

    runner_dict = {
        "get_rates": get_rates,
        "get_flow_rates": get_flow_rates,
        "get_comp_rates": get_comp_rates,
        "calc_initial_pop": calc_initial_pop,
        "get_compartment_infectiousness": get_compartment_infectiousness,
        "get_ode_solution": get_ode_solution,
        "calc_derived_outputs": calc_derived_outputs,
    }

    return run_model, runner_dict
