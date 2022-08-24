from __future__ import annotations
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from summer import CompartmentalModel

from computegraph import ComputeGraph
from computegraph.types import Data, Function
from computegraph.utils import invert_dict

from summer.parameters.param_impl import GraphObjectParameter, FloatParameter
from summer.adjust import Overwrite


def as_graph_object(obj):
    if isinstance(obj, GraphObjectParameter):
        return obj.obj
    elif isinstance(obj, FloatParameter):
        # FIXME:
        # We probably just want these to be floats in most cases,
        # to avoid cluttering up the graph
        # But we also need to make sure we get a GraphObject out at the
        # end of map_flow_keys
        return Data(obj.value)
    else:
        raise TypeError(obj)


def map_flow_keys(m: CompartmentalModel) -> dict:
    realised_flows = {}

    for i, f in enumerate(m._flows):
        full_flow = [as_graph_object(f.param)]
        for a in f.adjustments:
            if isinstance(a, Overwrite):
                full_flow = [as_graph_object(a.param)]
            else:
                full_flow.append(as_graph_object(a.param))
        out_func = full_flow[0]
        for fparam in full_flow[1:]:
            out_func = out_func * fparam
        f._realised = realised_flows[i] = GraphObjectParameter(out_func)

    return realised_flows


def build_model_graph(m: CompartmentalModel) -> dict:

    map_flow_keys(m)

    obj_table = {}
    flow_keys = {}

    def get_obj_key(obj, base_name, unique=False):
        if isinstance(obj, GraphObjectParameter):
            if obj.obj not in obj_table:
                if unique:
                    name = base_name
                    if name in obj_table.values():
                        raise KeyError("Object with name {name} already exists")
                else:
                    name = f"{base_name}_{len(obj_table)}"
                obj_table[obj.obj] = name
            return obj_table[obj.obj]
        # elif isinstance(obj, FloatParameter):
        #    print("No!")
        #    return get_obj_key(GraphObjectParameter(Data(obj.value)), base_name)
        else:
            raise TypeError(obj)

    for i, f in enumerate(m._flows):
        fpkey = get_obj_key(f._realised, f"{f.name}_rate")
        flow_keys[i] = fpkey
        f._graph_key = fpkey

    for s in m._stratifications:
        if s.mixing_matrix is not None:
            get_obj_key(s.mixing_matrix, f"{s.name}_mixing_matrix", True)

    for k, v in m._computed_values_graph_dict.items():
        name = f"computed_values.{k}"
        get_obj_key(GraphObjectParameter(v), name, True)

    # Capture computed values in dictionary so that
    # 'old-style' summer functions (t,cv) can use them
    def capture_kwargs(*args, **kwargs):
        return kwargs

    cv_func = Function(capture_kwargs, kwargs=m._computed_values_graph_dict)
    cv_func.node_name = "gather_cv"
    get_obj_key(GraphObjectParameter(cv_func), "computed_values", True)

    model_graph = invert_dict(obj_table)

    return ComputeGraph(model_graph)
