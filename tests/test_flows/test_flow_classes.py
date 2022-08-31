import numpy as np
import pytest

from summer2 import Compartment, adjust
from summer2.flows import (
    CrudeBirthFlow,
    DeathFlow,
    ImportFlow,
    InfectionDensityFlow,
    InfectionFrequencyFlow,
    ReplacementBirthFlow,
    TransitionFlow,
)


def test_transition_flow_get_net_flow():
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t, cv: 2 * t,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 2 * 3 * 7


def test_transition_flow_get_net_flow_with_adjust():
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t, cv: 2 * t,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 2 * 5 * 7 * 13


def test_import_flow_get_net_flow():
    flow = ImportFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t, cv: 0.1 * t,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 0.1 * 7


def test_import_flow_get_net_flow_with_adjust():
    flow = ImportFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t, cv: 0.1 * t,
        adjustments=[adjust.Multiply(13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 0.1 * 7 * 13


def test_crude_birth_flow_get_net_flow():
    flow = CrudeBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t, cv: 0.1 * t,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 0.1 * 7 * (1 + 3 + 5)


def test_crude_birth_flow_get_net_flow_with_adjust():
    flow = CrudeBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t, cv: 0.1 * t,
        adjustments=[adjust.Multiply(13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 0.1 * 7 * (1 + 3 + 5) * 13


def test_replace_deaths_birth_flow_get_net_flow():
    flow = ReplacementBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t, cv: 23,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 23


def test_replace_deaths_birth_flow_get_net_flow_with_adjust():
    flow = ReplacementBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t, cv: 23,
        adjustments=[adjust.Multiply(13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 23 * 13


def test_death_flow_get_net_flow():
    flow = DeathFlow(
        name="flow",
        source=Compartment("I"),
        param=lambda t, cv: 2 * t,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 2 * 3 * 7


def test_death_flow_get_net_flow_with_adjust():
    flow = DeathFlow(
        name="flow",
        source=Compartment("I"),
        param=lambda t, cv: 2 * t,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 2 * 5 * 7 * 13


@pytest.mark.parametrize("FlowClass", [InfectionDensityFlow, InfectionFrequencyFlow])
def test_infection_get_net_flow(FlowClass):
    flow = FlowClass(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t, cv: 2 * t,
        find_infectious_multiplier=lambda s, d: 23,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 2 * 3 * 7 * 23


@pytest.mark.parametrize("FlowClass", [InfectionDensityFlow, InfectionFrequencyFlow])
def test_infection_get_net_flow_with_adjust(FlowClass):
    flow = FlowClass(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t, cv: 2 * t,
        find_infectious_multiplier=lambda s, d: 23,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, {}, 7)
    assert net_flow == 2 * 3 * 7 * 23 * 13
