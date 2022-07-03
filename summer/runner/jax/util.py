from jax import jit
from .jax import build_get_rates

from jax.experimental import ode


def get_runner(m, parameters: dict):
    from summer.runner import VectorizedRunner

    runner = VectorizedRunner(m)
    runner.prepare_to_run(parameters=parameters)
    m._backend = runner
    return runner


def build_model_with_jax(build_func, parameters):
    m = build_func(use_jax=True)
    runner = get_runner(m, parameters)
    get_rates = jit(build_get_rates(runner))

    @jit
    def get_comp_rates(comp_vals, t, parameters):
        return get_rates(comp_vals, t, parameters)[0]

    @jit
    def get_jode_solution(parameters):
        return ode.odeint(get_comp_rates, m.initial_population, m.times, parameters)

    m_nojax = build_func(use_jax=False)

    return m_nojax, get_jode_solution
