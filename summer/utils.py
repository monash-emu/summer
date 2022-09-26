"""
General utility functions used accross submodules
"""

import pandas as pd
from datetime import timedelta, datetime
from typing import Iterable, Any

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

def get_model_param_value(
    param: Any, time: float, computed_values: dict, parameters: dict, mul_outputs=False
) -> Any:
    """Get the value of anything that might be possibly be used as as a parameter
    Variable, Function, list of parameters, callable, or just return a python object

    Args:
        param: Any
        time: Model supplied time
        computed_values: Model supplied computed values
        parameters: Parameters dictionary
        mul_outputs: If param is a list, return the product of its components

    Raises:
        Exception: Raised if param is a Variable with unknown source

    Returns:
        Any: Parameter output
    """
    if callable(param):
        return param(time, computed_values)
    elif isinstance(param, list):
        if mul_outputs:
            value = 1.0
            for subparam in param:
                value *= get_model_param_value(subparam, time, computed_values, parameters)
            return value
    else:
        return param