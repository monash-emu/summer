"""
This module contains classes which define adjustments to model parameters.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union

FlowParam = Union[float, Callable[[float], float]]


class BaseAdjustment(ABC):
    """
    :meta private:
    """

    def __init__(self, param: FlowParam):
        self.param = param

    @abstractmethod
    def get_new_value(self, value: float, input_values: dict, time: float) -> float:
        pass

    def _is_equal(self, adj):
        # Used for testing.
        return type(self) is type(adj) and self.param == adj.param

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.param}'>"


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

    def get_new_value(self, value: float, input_values: dict, time: float) -> float:
        """
        Returns the adjusted value for a given time.

        Args:
            value: The value to adjust.
            time: The time to be used for any time-varying functions.

        Returns:
            float: The new, adjusted value.

        """
        return self.param(time, input_values) * value if callable(self.param) else self.param * value


class Overwrite(BaseAdjustment):
    """
    Am overwrite-based adjustment of a parameter.
    The new parameter value is the supplied parameter, overwriting any previous values.

    Args:
        param: The parameter to be used in place of the unadjusted value.

    Examples:
        Create an adjustment to overwrite the previous value with 1.5::

            adjust.Overwrite(1.5)

        Create an adjustment to overwrite the previous value with the value of a time varying function::

            arbitrary_function = lambda time: 2 * time + 1
            adjust.Overwrite(arbitrary_function)

    """

    def get_new_value(self, value: float, input_values: dict, time: float) -> float:
        """
        Returns the adjusted value for a given time.

        Args:
            value: The value to adjust.
            time: The time to be used for any time-varying functions.

        Returns:
            float: The new, adjusted value.

        """
        return self.param(time, input_values) if callable(self.param) else self.param

class AdjustmentComponent:
    def __init__(self, system: str, data):
        """Adjustment is a component of an AdjustmentSystem
        The component does not compute a value directly, rather it contains
        the data that the system can use to compute all its values
        in a vectorized fashion

        Args:
            system (str): Name matching a system registered via add_adjustment_system
            data ([type]): Data of any type matching the system's interface 
        """
        self.system = system
        self.data = data

class AdjustmentSystem:
    def __init__(self):
        pass

    @abstractmethod
    def prepare_to_run(self, component_data: list):
        pass

    @abstractmethod
    def get_weights_at_time(self, time, input_values):
        pass