from __future__ import annotations
from ast import Mod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from summer import CompartmentalModel

class ModelVariable:
    def __init__(self, name: str, source: str = None):
        self.name = name
        self.source = source

    def __hash__(self):
        return hash((self.name, self.source))

    def __repr__(self):
        return f"ModelVariable {self.source}[{self.name}]"

    def __eq__(self, other):
        return (self.name == other.name) and (self.source == other.source)

class ModelParameter(ModelVariable):
    def __init__(self, name: str):
        super().__init__(name, "parameters")

    def __repr__(self):
        return f"Param {self.name}"

class ComputedValue(ModelVariable):
    def __init__(self, name: str):
        super().__init__(name, "computed_values")

    def __repr__(self):
        return f"ComputedValue {self.name}"

class DerivedOutput(ModelVariable):
    def __init__(self, name: str):
        super().__init__(name, "derived_outputs")

    def __repr__(self):
        return f"DerivedOutput {self.name}"

class ModelFunction:
    def __init__(self, func: callable, args: tuple = None, kwargs: dict = None):
        self.func = func
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if not (isinstance(args, tuple) or isinstance(args, list)):
            raise TypeError("Args must be list or tuple", args)
        self.args = tuple(args)
        self.kwargs = kwargs

    def __hash__(self):
        return hash((self.func, self.args, tuple(self.kwargs.items())))

    def __repr__(self):
        return f"ModelFunction: func={self.func}, args={self.args}, kwargs={self.kwargs})"

class TimeVaryingFunction(ModelFunction):
    def __repr__(self):
        return f"TimeVaryingFunction: func={self.func}, args={self.args}, kwargs={self.kwargs})"


def is_func(param) -> bool:
    """Wrapper to handle ModelFunction or callable

    Args:
        param: Flow or adjustment parameter

    Returns:
        bool: Is a function or function wrapper
    """

    return isinstance(param, ModelFunction) or callable(param)

def get_param_value(param, time, computed_values, parameters) -> float:
    if isinstance(param, ModelParameter):
        return parameters[param.name]
    elif isinstance(param, ComputedValue):
        return computed_values[param.name]
    elif isinstance(param, TimeVaryingFunction):
        sources = dict(computed_values=computed_values, parameters=parameters)
        args, kwargs = build_args(param.args, param.kwargs, sources)
        return param.func(time, *args, **kwargs)
    elif isinstance(param, ModelFunction):
        sources = dict(computed_values=computed_values, parameters=parameters)
        args, kwargs = build_args(param.args, param.kwargs, sources)
        return param.func(*args, **kwargs)
    elif callable(param):
        return param(time, computed_values)
    else:
        return param

def build_args(args: tuple, kwargs: dict, sources: dict):
    out_args = []
    for a in args:
        if isinstance(a, ModelVariable):
            out_args.append(sources[a.source][a.name])
        else:
            out_args.append(a)
    out_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, ModelVariable):
            out_kwargs[k] = sources[v.source][v.name]
        else:
            out_kwargs[k] = v
    return out_args, out_kwargs

def extract_params(obj):
    if isinstance(obj, ModelParameter):
        return [obj.name]
    elif isinstance(obj, ModelFunction):
        return [v.name for v in obj.kwargs.values() if isinstance(v, ModelParameter)]
    else:
        return []

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
        if params := extract_params(f.param):
            append_list(out_params, params, ("FlowParam", f))
            
    # We can actually skip this bit and get more intelligble results by
    # just looking at the stratifications
    # This one of those classic summer dualities
    #    for a in f.adjustments:
    #        params = extract_params(a.param)
    #        for p in params:
    #            append(out_params, p, ("Adjustment",f,a))
    
    
    # Inside stratifications - we have retained some useful information...
    for s in m._stratifications:
        # Flow adjustments live here quite happily
        # Flow _parameters_ however are stratified to oblivion, hence the section above ^^^^^
        for fname, adjustments in s.flow_adjustments.items():
            for adj, source_strata, dest_strata in adjustments:
                for k,v in adj.items():
                    if v is not None:
                        params = extract_params(v.param)
                        append_list(out_params, params, ("FlowAdjustment", fname, source_strata, dest_strata))
                        
        # Mixing matrices can be inspected here
        # They might sort of live in the model itself too...
        append_list(out_params, extract_params(s.mixing_matrix), ("MixingMatrix", s))
    
    # Derived outputs
    for k, req in m._derived_output_requests.items():
        if req['request_type'] == 'param_func':
            append_list(out_params, extract_params(req['func']), ("DerivedOutput", k))
            
    return out_params
