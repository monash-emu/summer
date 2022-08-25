from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

from summer.runner.vectorized_runner import VectorizedRunner

from .model_impl import build_run_model

if TYPE_CHECKING:
    from summer import CompartmentalModel


def get_runner(m: CompartmentalModel):
    m.finalize()
    runner = VectorizedRunner(m)
    runner.prepare_structural()
    m._backend = runner
    return runner


def build_model_with_jax(build_func: callable) -> Tuple[CompartmentalModel, callable]:
    """This exists primarily as a test shim and should not be considered the main entry point

    Args:
        build_func (callable): A build model function taking a use_jax boolean argument

    Returns:
        Tuple of non-Jax ComparmentalModel, and a (jittable) Jax callable
    """
    m = build_func(use_jax=True)
    runner = get_runner(m)

    run_model, runner_dict = build_run_model(runner)

    m_nojax = build_func(use_jax=False)

    return m_nojax, run_model
