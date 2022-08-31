from __future__ import annotations
from typing import TYPE_CHECKING, Any
from computegraph.types import Variable, Function, GraphObject, Data, build_args  # noqa: F401
from computegraph.utils import extract_variables, is_var

if TYPE_CHECKING:
    from summer2 import CompartmentalModel


class Parameter(Variable):
    def __init__(self, key: str):
        super().__init__(key, "parameters")

    def __repr__(self):
        return f"Parameter {self.key}"


class DerivedOutput(Variable):
    def __init__(self, name: str):
        super().__init__(name, "derived_outputs")

    def __repr__(self):
        return f"DerivedOutput {self.key}"


class ModelVariable(Variable):
    def __init__(self, name: str):
        super().__init__(name, "model_variables")

    def __repr__(self):
        return f"ModelVariable {self.key}"


CompartmentValues = ModelVariable("compartment_values")
ComputedValuesDict = Variable("computed_values", "graph_locals")
Time = ModelVariable("time")


def is_func(param) -> bool:
    """Wrapper to handle Function or callable

    Args:
        param: Flow or adjustment parameter

    Returns:
        bool: Is a function or function wrapper
    """
    from .param_impl import GraphObjectParameter

    return isinstance(param, Function) or callable(param)


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

    raise Exception("Catch")

    if isinstance(param, float):
        return param
    elif isinstance(param, GraphObject):
        sources = dict(
            computed_values=computed_values,
            parameters=parameters,
            model_variables={"time": time, "computed_values": computed_values},
        )
        return param.evaluate(**sources)
    elif isinstance(param, Variable):
        if param.source == "parameters":
            return parameters[param.key]
        elif param.source == "computed_values":
            return computed_values[param.key]
        else:
            raise Exception("Unsupported variable source", param, param.source)
    elif isinstance(param, Function):
        sources = dict(
            computed_values=computed_values, parameters=parameters, model_variables={"time": time}
        )
        args, kwargs = build_args(param.args, param.kwargs, sources)
        return param.func(*args, **kwargs)
    elif callable(param):
        return param(time, computed_values)
    elif (isinstance(param, list) or isinstance(param, tuple)) and mul_outputs:
        value = 1.0
        for subparam in param:
            value *= get_model_param_value(subparam, time, computed_values, parameters)
        return value
    else:
        return param


def get_static_param_value(obj: Any, static_graph_values: dict, passthrough=True) -> Any:
    """Get the value of a parameter, or of a function that depends only on parameters,
       or return obj if any other type

    Args:
        obj : The Variable, Function, or Python object
        parameters: Parameters dictionary
        passthrough: If True, will return unknown value types "as-is"; default is to raise TypeError

    Returns:
        The value of the object
    """
    from .param_impl import GraphObjectParameter

    # Might have some nested special classes
    if isinstance(obj, dict):
        return {
            k: get_static_param_value(v, static_graph_values, passthrough) for k, v in obj.items()
        }
    elif isinstance(obj, GraphObjectParameter):
        return static_graph_values[obj._graph_key]
    else:
        if passthrough:
            return obj
        else:
            raise TypeError(obj)
