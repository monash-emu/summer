"""
This module contains classes which define adjustments to model parameters.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union

from summer2.parameters import get_static_param_value
from summer2.parameters.param_impl import ModelParameter

FlowParam = Union[float, Callable[[float], float]]


class BaseAdjustment(ABC):
    """
    :meta private:
    """

    def __init__(self, param: FlowParam):
        self.param = param

    @abstractmethod
    def get_new_value(
        self, value: float, computed_values: dict, time: float, parameters: dict = None
    ) -> float:
        pass

    def _is_equal(self, adj):
        # Used for testing.
        return type(self) is type(adj) and self.param == adj.param

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.param}'>"

    def __hash__(self):
        return hash((type(self), self.param))

    def __eq__(self, o: object) -> bool:
        return self._is_equal(o)


class Multiply(BaseAdjustment):
    """
    A multiplication-based adjustment of a parameter.
    The new parameter value is the previous value, multiplied by the supplied parameter.

    Args:
        param: The parameter to be multiplied with the unadjusted value.


    Examples:
        Create an adjustment to multiply by a factor of 1.5::

            adjust.Multiply(1.5)

        Created an adjustment to multiply with the value of a time varying function::

            arbitrary_function = lambda time: 2 * time + 1
            adjust.Multiply(arbitrary_function)

    """

    def get_new_value(
        self, value: float, computed_values: dict, time: float, parameters: dict = None
    ) -> float:
        """
        Returns the adjusted value for a given time.

        Args:
            value: The value to adjust.
            time: The time to be used for any time-varying functions.

        Returns:
            float: The new, adjusted value.

        """
        # resolved_self = get_model_param_value(self.param, time, computed_values, parameters, True)
        if isinstance(self.param, ModelParameter):
            resolved_self = self.param.get_value(time, computed_values, parameters)
        elif callable(self.param):
            resolved_self = self.param(time, computed_values)
        else:
            resolved_self = self.param
        resolved_input = get_static_param_value(value, parameters, passthrough=True)
        return resolved_self * resolved_input


class Overwrite(BaseAdjustment):
    """
    Am overwrite-based adjustment of a parameter.
    The new parameter value is the supplied parameter, overwriting any previous values.

    Args:
        param: The parameter to be used in place of the unadjusted value.

    Examples:
        Create an adjustment to overwrite the previous value with 1.5::

            adjust.Overwrite(1.5)

        Create an adjustment to overwrite the previous value with the value of a
        time varying function::

            arbitrary_function = lambda time: 2 * time + 1
            adjust.Overwrite(arbitrary_function)

    """

    def get_new_value(
        self, value: float, computed_values: dict, time: float, parameters: dict = None
    ) -> float:
        """
        Returns the adjusted value for a given time.

        Args:
            value: The value to adjust.
            time: The time to be used for any time-varying functions.

        Returns:
            float: The new, adjusted value.

        """
        if isinstance(self.param, ModelParameter):
            return self.param.get_value(time, computed_values, parameters)
        elif callable(self.param):
            return self.param(time, computed_values)
        else:
            return self.param
        # return get_model_param_value(self.param, time, computed_values, parameters, True)


def enforce_wrapped(value, allowed, wrap):
    if any([isinstance(value, t) for t in allowed]):
        return value
    else:
        return wrap(value)


def enforce_multiply(value):
    return enforce_wrapped(value, [Multiply, Overwrite, type(None)], Multiply)
