from __future__ import annotations
from typing import TYPE_CHECKING

from summer.runner.jax.runner import JaxRunner

from .model_impl import build_run_model

if TYPE_CHECKING:
    from summer import CompartmentalModel


def get_runner(m: CompartmentalModel):
    runner = JaxRunner(m)
    runner.prepare_structural()
    m._backend = runner
    return runner


def build_model_with_jax(build_func: callable):
    m = build_func(use_jax=True)
    runner = get_runner(m)

    run_model = build_run_model(runner)

    m_nojax = build_func(use_jax=False)

    return m_nojax, run_model
