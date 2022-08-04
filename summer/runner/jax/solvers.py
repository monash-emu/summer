# flake8: noqa

from jax import lax, numpy as jnp


def euler(get_comp_rates, initial_population, times, model_params, model_data, timescale=1):

    times = jnp.linspace(times[0], times[-1], len(times) * timescale)
    timestep = 1.0 / timescale

    out_vals = jnp.empty((len(times), len(initial_population)))
    out_vals = out_vals.at[0].set(initial_population)

    max_i = len(times) - 1

    def body(state):
        i, out_vals = state
        t = times[i]
        comp_vals = out_vals[i]
        comp_rates = get_comp_rates(comp_vals, t, model_params, model_data)
        comp_vals = comp_vals + comp_rates * timestep
        out_vals = out_vals.at[i + 1].set(comp_vals)
        return i + 1, out_vals

    def cond(state):
        i, _ = state
        return i < max_i

    _, final = lax.while_loop(cond, body, (0, out_vals))
    return final


def rk4(get_comp_rates, initial_population, times, model_params, model_data):

    # times = jnp.linspace(times[0],times[-1],len(times) * timescale)
    # timestep = 1.0/timescale
    timestep = 1.0

    out_vals = jnp.empty((len(times), len(initial_population)))
    out_vals = out_vals.at[0].set(initial_population)

    max_i = len(times) - 1

    def body(state):
        i, out_vals = state
        t = times[i]
        comp_vals = out_vals[i]

        k1 = get_comp_rates(comp_vals, t, model_params, model_data)
        k2 = get_comp_rates(comp_vals + k1 / 2, t + timestep / 2, model_params, model_data)
        k3 = get_comp_rates(comp_vals + k2 / 2, t + timestep / 2, model_params, model_data)
        k4 = get_comp_rates(comp_vals + k3, t + timestep, model_params, model_data)
        comp_vals = comp_vals + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        out_vals = out_vals.at[i + 1].set(comp_vals)
        return i + 1, out_vals

    def cond(state):
        i, _ = state
        return i < max_i

    _, final = lax.while_loop(cond, body, (0, out_vals))
    return final
