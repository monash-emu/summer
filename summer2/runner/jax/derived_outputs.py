"""
Calculation of derived outputs.

A derived output is an additional output that the user has requested,
which can be calculated (or "derived") using the model results, which are the:

    - evaluation times
    - compartment sizes for each time
    - flow rates for each time

"""
import logging

from jax import numpy as jnp
import numpy as np

from computegraph import ComputeGraph
from computegraph.utils import get_relabelled_func
from computegraph.types import local

from summer2.parameters import Function, ModelVariable

logger = logging.getLogger()


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


def return_agg(*sources):
    return jnp.array(sources).sum(axis=0)


def build_aggregate_output(request):
    return Function(return_agg, [local(src) for src in request["sources"]])


def build_cumulative_output(request, name, times, baseline_offset=None):
    source_name = request["source"]
    start_time = request["start_time"]
    max_time = times.max()
    if start_time and start_time > max_time:
        # Handle case where the derived output starts accumulating after the last model timestep.
        msg = f"Cumulative output '{name}' start time {start_time} is greater than max model time {max_time}, defaulting to {max_time}"
        logger.warn(msg)
        start_time = max_time

    if baseline_offset is not None:
        raise NotImplementedError()

    if start_time is None:
        return Function(jnp.cumsum, [local(source_name)])
    else:
        assert start_time in times, f"Start time {start_time} not in times for '{name}'"
        start_idx = np.where(times == start_time)[0][0]

        def get_indexed_cumsum(in_arr):
            output = jnp.zeros(len(times), dtype=jnp.float64)
            output = output.at[start_idx:].set(jnp.cumsum(in_arr[start_idx:]))
            return output

        return Function(get_indexed_cumsum, [local(source_name)])


def build_function_output(request):
    func = request["func"]
    source_names = request["sources"]
    inputs = [local(s) for s in source_names]
    return Function(func, inputs)


def build_computed_value_output(request, name):
    return Function(lambda x: x[name], [ModelVariable("computed_values")])


def build_derived_outputs_runner(model):
    graph_dict = {}
    out_keys = []
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
        elif req_type == "agg":
            graph_dict[name] = build_aggregate_output(request)
        elif req_type == "cum":
            graph_dict[name] = build_cumulative_output(request, name, model.times)
        elif req_type == "func":
            graph_dict[name] = build_function_output(request)
        elif req_type == "computed_value":
            graph_dict[name] = build_computed_value_output(request, name)
        else:
            raise NotImplementedError(request)
        if request["save_results"]:
            out_keys.append(name)

    if model._derived_outputs_whitelist:
        out_keys = model._derived_outputs_whitelist

    cg = ComputeGraph(graph_dict)
    return cg.get_callable(targets=out_keys)
