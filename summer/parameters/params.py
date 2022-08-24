from __future__ import annotations
from typing import TYPE_CHECKING, Any
from computegraph.types import Variable, Function, GraphObject, Data, build_args  # noqa: F401
from computegraph.utils import extract_variables, is_var

if TYPE_CHECKING:
    from summer import CompartmentalModel


class Parameter(Variable):
    def __init__(self, key: str):
        super().__init__(key, "parameters")

    def __repr__(self):
        return f"Parameter {self.key}"


class ComputedValue(Variable):
    def __init__(self, name: str):
        super().__init__(name, "computed_values")

    def __repr__(self):
        return f"ComputedValue {self.key}"


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
ComputedValuesDict = ModelVariable("computed_values")
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


def get_static_param_value(obj: Any, parameters: dict, mul_outputs: bool = False) -> Any:
    """Get the value of a parameter, or of a function that depends only on parameters,
       or return obj if any other type

    Args:
        obj : The Variable, Function, or Python object
        parameters: Parameters dictionary

    Returns:
        The value of the object
    """
    from .param_impl import ModelParameter

    # Might have some nested special classes
    if isinstance(obj, dict):
        return {k: get_static_param_value(v, parameters) for k, v in obj.items()}
    elif isinstance(obj, ModelParameter):
        return obj.get_value(0.0, {}, parameters)
    elif isinstance(obj, float):
        return obj
    elif is_var(obj, "parameters"):
        return parameters[obj.key]
    elif isinstance(obj, Function):
        return obj.call(sources={"parameters": parameters})
    elif mul_outputs and (isinstance(obj, list) or isinstance(obj, tuple)):
        value = 1.0
        for subparam in obj:
            value *= get_static_param_value(subparam, parameters)
        return value
    else:
        return obj


def extract_params(obj):
    from .param_impl import GraphFunction, GraphParameter, CompoundParameter, LazyGraphParameter

    if isinstance(obj, GraphParameter):
        return [Parameter(obj.name)]
    elif isinstance(obj, LazyGraphParameter):
        return [Parameter(k) for k in obj.param_keys]
    elif isinstance(obj, CompoundParameter):
        out_params = []
        for sp in obj.subparams:
            out_params += extract_params(sp)
        return out_params
    elif isinstance(obj, GraphFunction):
        obj = obj.func
    if isinstance(obj, Function):
        out_params = []
        for a in obj.args:
            out_params += extract_params(a)
        for a in obj.kwargs.values():
            out_params += extract_params(a)
        return out_params
    return extract_variables(obj, source="parameters")


def find_all_parameters(m: CompartmentalModel):
    # Where could they hide?

    out_params = {}

    def append(target, key, value):
        if key not in out_params:
            target[key] = []
        target[key].append(value)

    def append_list(target, params, value):
        for p in params:
            append(target, p, value)

    # Inside flows
    for f in m._flows:
        params = extract_params(f.param)
        if params:
            append_list(out_params, params, ("FlowParam", f))

    # Initial population

    ipop_params = extract_params(m._init_pop_dist)
    append_list(out_params, ipop_params, ("InitialPopulation", m._init_pop_dist))

    # Inside stratifications - we have retained some useful information...
    for s in m._stratifications:
        params = extract_params(s.population_split)
        append_list(out_params, params, ("PopulationSplit", s, s.population_split))

        # Flow adjustments live here quite happily
        # Flow _parameters_ however are stratified to oblivion, hence the section above ^^^^^
        # for fname, adjustments in s.flow_adjustments.items():
        #    for adj, source_strata, dest_strata in adjustments:
        #        for k, v in adj.items():
        # if v is not None:
        #     params = extract_params(v.param)
        #     append_list(
        #         out_params,
        #         params,
        #         ("FlowAdjustment", fname, source_strata, dest_strata),
        #     )

        for comp, adjustments in s.infectiousness_adjustments.items():
            for strain, adjustment in adjustments.items():
                if adjustment is not None:
                    params = extract_params(adjustment.param)
                    append_list(
                        out_params,
                        params,
                        ("InfectiousnessAdjustment", s, comp, strain),
                    )
        # Mixing matrices can be inspected here
        # They might sort of live in the model itself too...
        append_list(out_params, extract_params(s.mixing_matrix), ("MixingMatrix", s))

    # Computed values
    for k, v in m._computed_values_graph_dict.items():
        params = extract_params(v)
        append_list(out_params, params, ("ComputedValue", k, v))

    # Derived outputs
    for k, req in m._derived_output_requests.items():
        if req["request_type"] == "param_func":
            append_list(out_params, extract_params(req["func"]), ("DerivedOutput", k))

    return out_params
