from __future__ import annotations
from typing import TYPE_CHECKING, List
from computegraph.types import Variable, Function, Parameter
from computegraph.utils import extract_variables

if TYPE_CHECKING:
    from summer import CompartmentalModel

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

class TimeVaryingFunction(Function):
    def __repr__(self):
        return f"TimeVaryingFunction: func={self.func}, args={self.args}, kwargs={self.kwargs})"

CompartmentValues = Variable("compartment_values", "model_variables")
Time = Variable("time", "model_variables")

def is_func(param) -> bool:
    """Wrapper to handle Function or callable

    Args:
        param: Flow or adjustment parameter

    Returns:
        bool: Is a function or function wrapper
    """

    return isinstance(param, Function) or callable(param)

def get_param_value(param, time, computed_values, parameters) -> float:
    if isinstance(param, Parameter):
        return parameters[param.name]
    elif isinstance(param, ComputedValue):
        return computed_values[param.name]
    #elif isinstance(param, TimeVaryingFunction):
    #    sources = dict(computed_values=computed_values, parameters=parameters)
    #    args, kwargs = build_args(param.args, param.kwargs, sources)
    #    return param.func(time, *args, **kwargs)
    elif isinstance(param, Function):
        sources = dict(computed_values=computed_values, parameters=parameters,model_variables={'time': time})
        args, kwargs = build_args(param.args, param.kwargs, sources)
        return param.func(*args, **kwargs)
    elif callable(param):
        return param(time, computed_values)
    else:
        return param

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
    return extract_variables(obj, source='parameters')

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
