from typing import Tuple, List, Iterable, Any
from numbers import Real

from computegraph.types import GraphObject, Data, Function
from computegraph.utils import defer

from summer.parameters.params import (
    build_args,
    is_var,
    Function,
    Variable,
    ComputedValue,
    ComputedValuesDict,
    Time,
)


class ModelParameter:
    def get_value(self, time: float, computed_values: dict, parameters: dict):
        raise NotImplementedError

    def __eq__(self, other):
        return hash(other) == hash(self)

    def is_time_varying(self):
        return False


class FloatParameter(ModelParameter):
    def __init__(self, value):
        self.value = value

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"FloatParameter: {self.value}"


class ComputedValueParameter(ModelParameter):
    def __init__(self, name):
        self.name = name

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return computed_values[self.name]

    def __hash__(self):
        return hash((self.name, "computed_values"))

    def __repr__(self):
        return f"ComputedValue: {self.name}"

    def is_time_varying(self):
        return True


class GraphFunction(ModelParameter):
    def __init__(self, func):
        self.func = func

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        sources = dict(
            computed_values=computed_values, parameters=parameters, model_variables={"time": time}
        )
        args, kwargs = build_args(self.func.args, self.func.kwargs, sources)
        return self.func.func(*args, **kwargs)

    def __hash__(self):
        return hash(self.func)

    def __repr__(self):
        return f"GraphFunction: {self.func}"

    def is_time_varying(self):
        has_time_var = Time in self.func.args or Time in self.func.kwargs.values()
        has_computed_value = any([isinstance(arg, ComputedValue) for arg in self.func.args])
        has_computed_value = has_computed_value or any(
            [isinstance(arg, ComputedValue) for arg in self.func.kwargs.values()]
        )

        return has_time_var or has_computed_value


class GraphObjectParameter(ModelParameter):
    def __init__(self, obj):
        self.obj = obj

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        sources = dict(
            computed_values=computed_values,
            parameters=parameters,
            model_variables={"time": time, "computed_values": computed_values},
        )
        return self.obj.evaluate(**sources)

    def __hash__(self):
        return hash(self.obj)

    def __eq__(self, other):
        return self.obj == other.obj

    def is_time_varying(self):
        return True


def get_modelparameter_from_param(param, allow_any=False) -> Tuple[ModelParameter, List[str]]:
    """Create a ModelParameter subclass object that can be called inside a model loop
    (ie time, computed_values and parameters are available)
    These are mostly flow params and adjustments
    Return this object, as well as a list of keys of all parameters found during its
    construction

    Args:
        param: Anything that might resolve to a single float
        allow_any: Whether to allow any generic data and return as Data

    Raises:
        TypeError: Raised if the input param cannot be resolved

    Returns:
        The resolved ModelParameter, along with any parameter keys encountered in the resolution
    """
    if isinstance(param, ModelParameter):
        # We've already transformed this parameter
        return param
    elif isinstance(param, GraphObject):
        return GraphObjectParameter(param)
    elif isinstance(param, Real):
        return GraphObjectParameter(Data(param))
    elif callable(param):
        return GraphObjectParameter(defer(param)(Time, ComputedValuesDict))
    elif isinstance(param, ComputedValue):
        raise Exception("Really?")
        return ComputedValueParameter(param.key), []
    else:
        if allow_any:
            return GraphObjectParameter(Data(param))
        else:
            raise TypeError(f"Unsupported model parameter type {type(param)}", param)


def get_reparameterized_dict(d):
    return {k: get_modelparameter_from_param(v) for k, v in d.items()}


def finalize_parameters(model):
    """Called as part of model.finalize
    This ensures all parameters (and function calls) have concrete computegraph.Variable
    realisations.  We can scan all possible parameterized sites, and replace params
    with appropriate types, as well as collecting all referenced named parameters

    Args:
        model: The CompartmentalModel to finalize
    """

    all_params = []

    # Flow parameters and adjustments
    for f in model._flows:
        f.param = get_modelparameter_from_param(f.param)

        for adj in f.adjustments:
            if adj is not None:
                adj.param = get_modelparameter_from_param(adj.param)

    # Initial population
    model._init_pop_dist = get_reparameterized_dict(model._init_pop_dist)

    # Stratifications - population split, infectiousness adjustments, mixing matrix
    for s in model._stratifications:
        s.population_split = get_reparameterized_dict(s.population_split)

        for comp, adjustments in s.infectiousness_adjustments.items():
            for strain, adjustment in adjustments.items():
                if adjustment is not None:
                    param = get_modelparameter_from_param(adjustment.param)
                    adjustment.param = param

        if s.mixing_matrix is not None:
            param = get_modelparameter_from_param(s.mixing_matrix, True)
            s.mixing_matrix = param

    # Computed values
    # cv_graph = {}

    # for k, v in model._computed_values_graph_dict.items():
    #    param, pkeys = concretize_function_args(v, builder)
    #    cv_graph[k] = param
    #    all_params += pkeys

    # model._computed_values_graph_dict = cv_graph

    # Derived outputs
    # for k, v in model._derived_output_requests.items():
    #    req_type = v["request_type"]
    #    if req_type == "param_func":
    #        param, pkeys = concretize_function_args(v["func"], builder)
    #        v["func"] = param
    #        all_params += pkeys
