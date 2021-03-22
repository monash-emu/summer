import numpy as np
import pytest

from summer.legacy.compartment import Compartment
from summer.legacy.constants import FlowAdjustment
from summer.legacy.flow import InfectionDeathFlow, UniversalDeathFlow
from summer.legacy.stratification import Stratification


@pytest.mark.parametrize("FlowClass", [InfectionDeathFlow, UniversalDeathFlow])
def test_death_flow_get_net_flow(FlowClass):
    flow = FlowClass(
        source=Compartment("infect"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7


@pytest.mark.parametrize("FlowClass", [InfectionDeathFlow, UniversalDeathFlow])
def test_death_flow_get_net_flow_with_adjust(FlowClass):
    flow = FlowClass(
        source=Compartment("infect"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[(FlowAdjustment.MULTIPLY, 13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 5 * 7 * 13


def _get_param_value(name, time):
    return 2 * time
