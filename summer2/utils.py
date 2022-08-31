"""
General utility functions used accross submodules
"""

import pandas as pd
from datetime import timedelta, datetime
from typing import Iterable
import numpy as np


def get_scenario_start_index(base_times, start_time):
    """
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    assert (
        base_times[0] <= start_time
    ), f"Scenario start time {start_time} is before baseline has started"
    indices_after_start_index = [idx for idx, time in enumerate(base_times) if time > start_time]
    if not indices_after_start_index:
        raise ValueError(f"Scenario start time {start_time} is set after the baseline time range")

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index


def ref_times_to_dti(ref_date: datetime, times: Iterable) -> pd.DatetimeIndex:
    """Return a DatetimeIndex whose values after ref_date offset by times

    Args:
        ref_date (datetime): The reference  date ('epoch') from which offsets are computed
        times (Iterable): Offsets to ref_date (int or float)

    Returns:
        (pd.DatetimeIndex): Index suitable for constructing Series or DataFrame
    """

    return pd.DatetimeIndex([ref_date + timedelta(t) for t in times])


def clean_compartment_values(compartment_values: np.ndarray):
    """
    Zero out -ve compartment sizes in flow rate calculations,
    to prevent negative values from messing up the direction of flows.
    We don't expect large -ve values, but there can be small ones due to numerical errors.
    """
    comp_vals = compartment_values.copy()
    zero_mask = comp_vals < 0
    comp_vals[zero_mask] = 0
    return comp_vals
