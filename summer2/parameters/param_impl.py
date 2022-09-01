from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from summer2 import CompartmentalModel

from functools import partial
from typing import Tuple, List, Iterable, Any
from numbers import Real

from computegraph.types import GraphObject, Data, Function, Variable
from computegraph.utils import defer, invert_dict, assign
from computegraph import ComputeGraph
from computegraph.jaxify import get_modules

fnp = get_modules()["numpy"]

import numpy as np

from summer2.parameters.params import (
    build_args,
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


class _ComputedValueParameter(ModelParameter):
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


class _GraphFunction(ModelParameter):
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
    else:
        if allow_any:
            return GraphObjectParameter(Data(param))
        else:
            raise TypeError(f"Unsupported model parameter type {type(param)}", param)


def get_reparameterized_dict(d):
    return {k: get_modelparameter_from_param(v) for k, v in d.items()}


def map_flow_keys(m: CompartmentalModel) -> dict:

    from summer2.adjust import Overwrite

    realised_flows = {}

    for i, f in enumerate(m._flows):
        full_flow = [f.param.obj]
        for a in f.adjustments:
            if isinstance(a, Overwrite):
                full_flow = [a.param.obj]
            else:
                full_flow.append(a.param.obj)
        out_func = full_flow[0]
        for fparam in full_flow[1:]:
            out_func = out_func * fparam
        realised_flows[i] = GraphObjectParameter(out_func)

    return realised_flows

def register_object_key(obj_table, obj, base_name, unique=False):
    if isinstance(obj, dict):
        for k, v in obj.items():
            register_object_key(obj_table, v, f"{base_name}_{k}")
    elif isinstance(obj, GraphObjectParameter):
        if obj.obj not in obj_table:
            if unique:
                name = base_name
                if name in obj_table.values():
                    raise KeyError("Object with name {name} already exists")
            else:
                if base_name in obj_table.values():
                    name = f"{base_name}_{len(obj_table)}"
                else:
                    name = base_name
            obj_table[obj.obj] = name
        obj_key = obj_table[obj.obj]
        obj._graph_key = obj_key
        return obj_key
    else:
        raise TypeError(obj, base_name)

def finalize_parameters(model):
    """Called as part of model.finalize
    This ensures all parameters (and function calls) have concrete computegraph.Variable
    realisations.  We can scan all possible parameterized sites, and replace params
    with appropriate types, as well as collecting all referenced named parameters

    Args:
        model: The CompartmentalModel to finalize
    """

    obj_table = {}

    register_obj_key = partial(register_object_key, obj_table)

    # Flow parameters and adjustments
    for f in model._flows:
        f.param = get_modelparameter_from_param(f.param)

        for adj in f.adjustments:
            if adj is not None:
                adj.param = get_modelparameter_from_param(adj.param)

    realised_flows = map_flow_keys(model)

    all_flow_keys = {}

    for i, f in enumerate(model._flows):
        fpkey = register_obj_key(realised_flows[i], f"{f.name}_rate")
        f._graph_key = fpkey
        if fpkey not in all_flow_keys:
            all_flow_keys[fpkey] = []
        key_store = all_flow_keys[fpkey]
        key_store.append(i)

    model._flow_key_map = {k: fnp.array(v, dtype=int) for k, v in all_flow_keys.items()}

    # Initial population
    if not hasattr(model, "_init_pop_dist"):
        raise Exception("Model initial population must be set before finalizing")

    model._init_pop_dist = get_reparameterized_dict(model._init_pop_dist)
    register_obj_key(model._init_pop_dist, "init_pop_dist")

    # Keep track of matrices so we can Kron them together later
    mixing_matrices = []
    # Stratifications - population split, infectiousness adjustments, mixing matrix
    for s in model._stratifications:
        s.population_split = get_reparameterized_dict(s.population_split)
        register_obj_key(s.population_split, f"{s.name}_pop_split")

        for comp, adjustments in s.infectiousness_adjustments.items():
            for stratum, adjustment in adjustments.items():
                if adjustment is not None:
                    param = get_modelparameter_from_param(adjustment.param)
                    adjustment.param = param
                    register_obj_key(adjustment.param, f"{s.name}_iadj_{comp}_{stratum}", True)

        if s.mixing_matrix is not None:
            param = get_modelparameter_from_param(s.mixing_matrix, True)
            s.mixing_matrix = param
            matrix_key = f"{s.name}_mixing_matrix"
            register_obj_key(s.mixing_matrix, matrix_key, True)
            mixing_matrices.append(param.obj)

    if len(mixing_matrices) == 0:
        param = get_modelparameter_from_param(Data(fnp.array([[1.0]])))
        register_obj_key(param, "mixing_matrix", True)
        model.mixing_matrix = param
    elif len(mixing_matrices) == 1:
        mm = mixing_matrices[0]
        param = get_modelparameter_from_param(defer(assign)(mm))
        register_obj_key(param, "mixing_matrix", True)
        model.mixing_matrix = param
    else:

        def compute_final_matrix(base_matrix, *args):
            cur_matrix = base_matrix
            for m in args:
                cur_matrix = fnp.kron(cur_matrix, m)
            return cur_matrix

        final_mat_func = defer(compute_final_matrix)(*mixing_matrices)
        param = get_modelparameter_from_param(final_mat_func)
        model.mixing_matrix = param
        register_obj_key(param, "mixing_matrix", True)

    for k, v in model._computed_values_graph_dict.items():
        name = f"computed_values.{k}"
        register_obj_key(GraphObjectParameter(v), name, True)

    # Capture computed values in dictionary so that
    # 'old-style' summer functions (t,cv) can use them
    def capture_kwargs(*args, **kwargs):
        return kwargs

    cv_func = Function(capture_kwargs, kwargs=model._computed_values_graph_dict)
    cv_func.node_name = "gather_cv"
    register_obj_key(GraphObjectParameter(cv_func), "computed_values", True)

    do_table = {}
    register_do_obj_key = partial(register_object_key, do_table)

    # Derived outputs
    for k, v in model._derived_output_requests.items():
        req_type = v["request_type"]
        if req_type == "param_func":
            name = f"derived_outputs.{k}"
            register_do_obj_key(GraphObjectParameter(v["func"]), name, True)

    do_graph = invert_dict(do_table)
    model._do_tracker_graph = ComputeGraph(do_graph, validate_keys=False)

    model_graph = invert_dict(obj_table)

    model.graph = ComputeGraph(model_graph, validate_keys=False)

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
