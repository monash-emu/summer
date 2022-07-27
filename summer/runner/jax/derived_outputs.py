"""
Calculation of derived outputs.

A derived output is an additional output that the user has requested,
which can be calculated (or "derived") using the model results, which are the:

    - evaluation times
    - compartment sizes for each time
    - flow rates for each time

"""
from jax import numpy as jnp

from computegraph import ComputeGraph
from computegraph.utils import get_relabelled_func

from summer.parameters import Function, ModelVariable


def build_flow_output(request, name, times, model_flows, idx_cache=None):

    flow_indices = []
    for flow_idx, flow in enumerate(model_flows):
        is_matching_flow = (
            flow.name == request["flow_name"]
            and ((not flow.source) or flow.source.has_strata(request["source_strata"]))
            and ((not flow.dest) or flow.dest.has_strata(request["dest_strata"]))
        )
        if is_matching_flow:
            flow_indices.append(flow_idx)

    flow_indices = jnp.array(flow_indices)

    use_raw_results = request["raw_results"]

    if use_raw_results:

        def get_flow_output(flows):
            return flows[:, flow_indices].sum(axis=1)

    else:

        def get_flow_output(flows):
            flow_vals = flows[:, flow_indices].sum(axis=1)
            midpoint_output = jnp.zeros(times.shape)
            midpoint_output = midpoint_output.at[0].set(flow_vals[0])
            interp_vals = (flow_vals[1:] + flow_vals[:-1]) * 0.5
            midpoint_output = midpoint_output.at[1:].set(interp_vals)
            return midpoint_output

    return Function(get_flow_output, [ModelVariable("flows")])


def build_compartment_output(request, name, compartments):

    req_compartments = request["compartments"]
    strata = request["strata"]
    comps = ((i, c) for i, c in enumerate(compartments) if c.has_name_in_list(req_compartments))
    indices = [i for i, c in comps if c.is_match(c.name, strata)]

    def summed_compartment_outputs(outputs):
        return outputs[:, indices].sum(axis=1)

    return Function(summed_compartment_outputs, [ModelVariable("outputs")])


def build_derived_outputs_runner(model):
    graph_dict = {}
    for name, request in model._derived_output_requests.items():
        req_type = request["request_type"]
        if req_type == "comp":
            graph_dict[name] = build_compartment_output(request, name, model.compartments)
        elif req_type == "param_func":
            graph_dict[name] = get_relabelled_func(
                request["func"], "derived_outputs", "graph_locals"
            )
        elif req_type == "flow":
            graph_dict[name] = build_flow_output(request, name, model.times, model._flows)
        else:
            raise NotImplementedError(request)
    cg = ComputeGraph(graph_dict)
    return cg.get_callable(False, False)
