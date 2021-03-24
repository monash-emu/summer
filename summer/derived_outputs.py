"""
Calculation of derived outputs.

A derived output is an additional output that the user has requested,
which can be calculated (or "derived") using the model results, which are the:

    - evaluation times
    - 

"""
import logging
from typing import Callable, Dict, List, Optional

import networkx
import numpy as np

from summer.compartment import Compartment
from summer.flows import BaseFlow

logger = logging.getLogger()


class DerivedOutputRequest:
    FLOW = "flow"
    COMPARTMENT = "comp"
    AGGREGATE = "agg"
    CUMULATIVE = "cum"
    FUNCTION = "func"


def calculate_derived_outputs(
    requests: List[dict],
    graph: networkx.DiGraph,
    outputs: np.ndarray,
    times: np.ndarray,
    timestep: float,
    flows: List[BaseFlow],
    compartments: List[Compartment],
    get_flow_rates: Callable[[np.ndarray, float], np.ndarray],
    whitelist: Optional[List[str]],
) -> Dict[str, np.ndarray]:
    """
    Calculates all requested derived outputs from the calculated compartment sizes.

    Args:
        requests: Descriptions of the new outputs we should create.
        graph: A DAG describing how the requests depend on each other.
        outputs: The compartmental model outputs - compartment sizes over time.
        times: The times that the outputs correspond to.
        timestep: The timestep used to generate the model times.
        flows: The flows used by the model.
        compartments: The compartments used by the model.
        get_flow_rates: A function that gets the model flow rates for a given state and time.
        whitelist: An optional subset of requests to evaluate.

    Returns:
        Dict[str, np.ndarray]: The timeseries results for each requested output.

    """
    assert outputs is not None, "Cannot calculate derived outputs: model has not been run."
    error_msg = "Cannot calculate derived outputs: dependency graph has cycles."
    assert networkx.is_directed_acyclic_graph(graph), error_msg
    graph = graph.copy()  # We're going to mutate the graph so copy it first.

    if whitelist:
        # Only calculate the required outputs and their dependencies, ignore everything else.
        required_nodes = set()
        for name in whitelist:
            # Find a list of the output required and its dependencies.
            output_dependencies = networkx.dfs_tree(graph.reverse(), source=name).reverse()
            required_nodes = required_nodes.union(output_dependencies.nodes)

        # Remove any nodes that aren't required from the graph.
        nodes = list(graph.nodes)
        for node in nodes:
            if not node in required_nodes:
                graph.remove_node(node)

    derived_outputs = {}
    outputs_to_delete_after = []

    # Calculate all flow rates and store in `flow_values` so that we can fulfill flow rate requests.
    # We need to do this here because some solvers do not evaluate all timesteps.
    flow_values = np.zeros((len(times), len(flows)))
    for time_idx, time in enumerate(times):
        # Flow rates are per unit time so we need to normalize by timestep.
        flow_values[time_idx, :] = get_flow_rates(outputs[time_idx], time) * timestep

    # Convert tracked flow values into a matrix where the 1st dimension is flow type, 2nd is time
    flow_values = np.array(flow_values).T

    # Calculate all the outputs in the correct order so that each output has its dependencies fulfilled.
    for name in networkx.topological_sort(graph):
        request = requests[name]
        request_type = request["request_type"]
        output = np.zeros(times.shape)

        if not request["save_results"]:
            # Delete the results of this output once the calcs are done.
            outputs_to_delete_after.append(name)

        if request_type == DerivedOutputRequest.FLOW:
            # User wants to track a set of flow rates over time.
            output = _get_flow_output(request, times, flows, flow_values)
        elif request_type == DerivedOutputRequest.COMPARTMENT:
            # User wants to track a set of compartment sizes over time.
            output = _get_compartment_output(request, outputs, compartments)
        elif request_type == DerivedOutputRequest.AGGREGATE:
            # User wants to track the sum of a set of outputs over time.
            output = _get_aggregate_output(request, derived_outputs)
        elif request_type == DerivedOutputRequest.CUMULATIVE:
            # User wants to track cumulative value of an output over time.
            output = _get_cumulative_output(request, name, times, derived_outputs)
        elif request_type == DerivedOutputRequest.FUNCTION:
            # User wants to track the results of a function of other outputs over time.
            output = _get_func_output(request, derived_outputs)

        derived_outputs[name] = output

    # Delete any intermediate outputs that we don't want to save.
    for name in outputs_to_delete_after:
        del derived_outputs[name]

    return derived_outputs


def _get_flow_output(request, times, flows, flow_values):
    this_flow_values = np.zeros_like(times)
    for flow_idx, flow in enumerate(flows):
        is_matching_flow = (
            flow.name == request["flow_name"]
            and ((not flow.source) or flow.source.has_strata(request["source_strata"]))
            and ((not flow.dest) or flow.dest.has_strata(request["dest_strata"]))
        )
        if is_matching_flow:
            this_flow_values += flow_values[flow_idx]

    use_raw_results = request["raw_results"]
    if use_raw_results:
        # Use interpolated flow rates wiuth no post-processing.
        return this_flow_values
    else:
        # Set the "flow rate" at time `t` to be an estimate of the flow rate
        # that is calculated at time `t-1`. By convention, flows are zero at t=0.
        # This is done so that we can estimate the number of people moving between compartments
        # using tracked flow rates.
        ignore_first_timestep_output = np.zeros(times.shape)
        ignore_first_timestep_output[1:] = this_flow_values[1:]
        offset_output = np.zeros(times.shape)
        offset_output[1:] = this_flow_values[:-1]
        return (offset_output + ignore_first_timestep_output) / 2


def _get_compartment_output(request, outputs, compartments):
    req_compartments = request["compartments"]
    strata = request["strata"]
    comps = ((i, c) for i, c in enumerate(compartments) if c.has_name_in_list(req_compartments))
    idxs = [i for i, c in comps if c.is_match(c.name, strata)]
    return outputs[:, idxs].sum(axis=1)


def _get_aggregate_output(request, derived_outputs):
    source_names = request["sources"]
    return sum([derived_outputs[s] for s in source_names])


def _get_cumulative_output(request, name, times, derived_outputs):
    source_name = request["source"]
    start_time = request["start_time"]
    max_time = times.max()
    if start_time and start_time > max_time:
        # Handle case where the derived output starts accumulating after the last model timestep.
        msg = f"Cumulative output '{name}' start time {start_time} is greater than max model time {max_time}, defaulting to {max_time}"
        logger.warn(msg)
        start_time = max_time

    if start_time is None:
        output = np.cumsum(derived_outputs[source_name])
    else:
        assert start_time in times, f"Start time {start_time} not in times for '{name}'"
        start_idx = np.where(times == start_time)[0][0]
        output = np.zeros_like(times)
        output[start_idx:] = np.cumsum(derived_outputs[source_name][start_idx:])

    return output


def _get_func_output(request, derived_outputs):
    func = request["func"]
    source_names = request["sources"]
    inputs = [derived_outputs[s] for s in source_names]
    return func(*inputs)
