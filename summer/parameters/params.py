from __future__ import annotations
from typing import TYPE_CHECKING, Any
from computegraph.types import Variable, Function, Data  # noqa: F401
from computegraph.utils import extract_variables, is_var

if TYPE_CHECKING:
    from summer import CompartmentalModel


class Parameter(Variable):
    def __init__(self, name: str):
        super().__init__(name, "parameters")

    def __repr__(self):
        return f"Parameter {self.name}"


class ComputedValue(Variable):
    def __init__(self, name: str):
        super().__init__(name, "computed_values")

    def __repr__(self):
        return f"ComputedValue {self.name}"


class DerivedOutput(Variable):
    def __init__(self, name: str):
        super().__init__(name, "derived_outputs")

    def __repr__(self):
        return f"DerivedOutput {self.name}"


class ModelVariable(Variable):
    def __init__(self, name: str):
        super().__init__(name, "model_variables")

    def __repr__(self):
        return f"DerivedOutput {self.name}"


CompartmentValues = ModelVariable("compartment_values")
Time = ModelVariable("time")


def is_func(param) -> bool:
    """Wrapper to handle Function or callable

    Args:
        param: Flow or adjustment parameter

    Returns:
        bool: Is a function or function wrapper
    """

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
    if isinstance(param, Variable):
        if param.source == "parameters":
            return parameters[param.name]
        elif param.source == "computed_values":
            return computed_values[param.name]
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
    elif isinstance(param, list):
        if mul_outputs:
            value = 1.0
            for subparam in param:
                value *= get_model_param_value(subparam, time, computed_values, parameters)
            return value

    else:
        return param


def get_static_param_value(obj: Any, parameters: dict) -> Any:
    """Get the value of a parameter, or of a function that depends only on parameters,
       or return obj if any other type

    Args:
        obj : The Variable, Function, or Python object
        parameters: Parameters dictionary

    Returns:
        The value of the object
    """
    if is_var(obj, "parameters"):
        return parameters[obj.name]
    elif isinstance(obj, Function):
        return obj.call(sources={"parameters": parameters})
    else:
        return obj


def build_args(args: tuple, kwargs: dict, sources: dict):
    out_args = []
    for a in args:
        if isinstance(a, Variable):
            out_args.append(sources[a.source][a.name])
        else:
            out_args.append(a)
    out_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Variable):
            out_kwargs[k] = sources[v.source][v.name]
        else:
            out_kwargs[k] = v
    return out_args, out_kwargs


def extract_params(obj):
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

    ipop_params = extract_params(m._initial_population_distribution)
    append_list(out_params, ipop_params, ("InitialPopulation", m._initial_population_distribution))

    # Inside stratifications - we have retained some useful information...
    for s in m._stratifications:
        params = extract_params(s.population_split)
        append_list(out_params, params, ("PopulationSplit", s, s.population_split))

        # Flow adjustments live here quite happily
        # Flow _parameters_ however are stratified to oblivion, hence the section above ^^^^^
        for fname, adjustments in s.flow_adjustments.items():
            for adj, source_strata, dest_strata in adjustments:
                for k, v in adj.items():
                    if v is not None:
                        params = extract_params(v.param)
                        append_list(
                            out_params,
                            params,
                            ("FlowAdjustment", fname, source_strata, dest_strata),
                        )

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
