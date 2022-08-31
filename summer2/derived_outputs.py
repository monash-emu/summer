"""
Calculation of derived outputs.

A derived output is an additional output that the user has requested,
which can be calculated (or "derived") using the model results, which are the:

    - evaluation times
    - compartment sizes for each time
    - flow rates for each time

"""
import logging
from typing import Callable, Dict, List, Optional

import networkx
import numpy as np

import pandas as pd

from summer2.compartment import Compartment
from summer2.flows import BaseFlow
from summer2.utils import get_scenario_start_index

from summer2.parameters import build_args

logger = logging.getLogger()


class DerivedOutputRequest:
    FLOW = "flow"
    COMPARTMENT = "comp"
    AGGREGATE = "agg"
    CUMULATIVE = "cum"
    FUNCTION = "func"
    COMPUTED_VALUE = "computed_value"
    PARAM_FUNCTION = "param_func"


def calculate_derived_outputs(
    requests: List[dict],
    graph: networkx.DiGraph,
    outputs: np.ndarray,
    times: np.ndarray,
    timestep: float,
    flows: List[BaseFlow],
    compartments: List[Compartment],
    get_flow_rates: Callable[[np.ndarray, float], np.ndarray],
    model,
    whitelist: Optional[List[str]],
    baseline=None,
    idx_cache=None,
    parameters: dict = None,
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
        baseline: Optional CompartmentalModel object to be used as a reference

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

    # If we have a baseline for comparison, get some basic information re offsets
    if baseline:
        baseline_start_index = get_scenario_start_index(baseline.times, times[0])

    # Calculate all flow rates and store in `flow_values` so that we can fulfil flow rate requests.
    # We need to do this here because some solvers do not necessarily evaluate all timesteps.
    flow_values = np.zeros((len(times), len(flows)))

    # These are 'extra' values computed by requested processes, and need to be tracked separately
    computed_values = []

    # FIXME: Another question for Matt - has my changes to the time requests stuffed this up?
    # Because the timestep for the last time interval can now be different from the earlier ones.
    # So do we need to assert that the duration is an exact multiple of the timestep?
    # Could cause silent problems, because presumably we have previously been specifying durations as multiples of the timestep.
    for time_idx, time in enumerate(times):
        # Flow rates are instantaneous; we need to provide an integrated value over timestep
        flow_rates_t, computed_values_t = get_flow_rates(outputs[time_idx], time)
        flow_values[time_idx, :] = flow_rates_t * timestep
        # Collect these as lists then build DataFrames afterwards
        computed_values.append(computed_values_t)

    # Collate list values into DataFrames
    computed_values = pd.DataFrame(
        columns=model.get_computed_value_keys(), data=computed_values, index=times
    )

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
            output, idx_cache = _get_flow_output(
                request, name, times, flows, flow_values, idx_cache
            )
        elif request_type == DerivedOutputRequest.COMPARTMENT:
            # User wants to track a set of compartment sizes over time.
            output, idx_cache = _get_compartment_output(
                request, name, outputs, compartments, idx_cache
            )
        elif request_type == DerivedOutputRequest.AGGREGATE:
            # User wants to track the sum of a set of outputs over time.
            output = _get_aggregate_output(request, derived_outputs)
        elif request_type == DerivedOutputRequest.CUMULATIVE:
            # User wants to track cumulative value of an output over time.
            if baseline:
                baseline_offset = baseline.derived_outputs[name][baseline_start_index]
            else:
                baseline_offset = None
            output = _get_cumulative_output(request, name, times, derived_outputs, baseline_offset)
        elif request_type == DerivedOutputRequest.FUNCTION:
            # User wants to track the results of a function of other outputs over time.
            output = _get_func_output(request, derived_outputs)
        elif request_type == DerivedOutputRequest.PARAM_FUNCTION:
            output = _get_param_func_output(request, derived_outputs, computed_values, parameters)
        # FIXME DerivedValue and InputValue should probably be combined
        elif request_type == DerivedOutputRequest.COMPUTED_VALUE:
            output = _get_computed_value_output(request, computed_values)

        derived_outputs[name] = output

    # Delete any intermediate outputs that we don't want to save.
    for name in outputs_to_delete_after:
        del derived_outputs[name]

    return derived_outputs, idx_cache


def _get_flow_output(request, name, times, flows, flow_values, idx_cache=None):
    this_flow_values = np.zeros_like(times)

    if not idx_cache:
        idx_cache = {}

    if not name in idx_cache:
        idx_cache[name] = []
        for flow_idx, flow in enumerate(flows):
            is_matching_flow = (
                flow.name == request["flow_name"]
                and ((not flow.source) or flow.source.has_strata(request["source_strata"]))
                and ((not flow.dest) or flow.dest.has_strata(request["dest_strata"]))
            )
            if is_matching_flow:
                idx_cache[name].append(flow_idx)

    for flow_idx in idx_cache[name]:
        this_flow_values += flow_values[flow_idx]

    use_raw_results = request["raw_results"]
    if use_raw_results:
        # Use interpolated flow rates with no post-processing.
        return this_flow_values, idx_cache
    else:
        # Set the "flow value" at time `t` to be a midpoint approximation of the integrated flow
        # bewteen `t-1` and 't'
        # This is done so that we can estimate the number of people moving between compartments
        # using tracked flow rates.
        midpoint_output = np.zeros(times.shape)
        # First point just uses existing value since no 't=-1' step is available
        # Client using these outputs will typically the first value
        midpoint_output[0] = this_flow_values[0]
        midpoint_output[1:] = (this_flow_values[1:] + this_flow_values[:-1]) * 0.5
        return midpoint_output, idx_cache


def _get_compartment_output(request, name, outputs, compartments, idx_cache=None):
    if not idx_cache:
        idx_cache = {}

    if not name in idx_cache:
        req_compartments = request["compartments"]
        strata = request["strata"]
        comps = ((i, c) for i, c in enumerate(compartments) if c.has_name_in_list(req_compartments))
        idx_cache[name] = [i for i, c in comps if c.is_match(c.name, strata)]

    idxs = idx_cache[name]
    return outputs[:, idxs].sum(axis=1), idx_cache


def _get_aggregate_output(request, derived_outputs):
    source_names = request["sources"]
    return sum([derived_outputs[s] for s in source_names])


def _get_cumulative_output(request, name, times, derived_outputs, baseline_offset=None):
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

    if baseline_offset:
        output += baseline_offset

    return output


def _get_func_output(request, derived_outputs):
    func = request["func"]
    source_names = request["sources"]
    inputs = [derived_outputs[s] for s in source_names]
    return func(*inputs)


def _get_param_func_output(request, derived_outputs, computed_values, parameters):
    mfunc = request["func"]
    sources = dict(
        derived_outputs=derived_outputs, parameters=parameters, computed_values=computed_values
    )
    args, kwargs = build_args(mfunc.args, mfunc.kwargs, sources)
    return mfunc.func(*args, **kwargs)


def _get_computed_value_output(request, computed_values):
    name = request["name"]
    return computed_values[name].to_numpy(dtype=float)


def get_scenario_start_index(base_times, start_time):
    """
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    assert (
        base_times[0] <= start_time
    ), f"Scenario start time {start_time} is before baseline has started"
    indices_after_start_index = [idx for idx, time in enumerate(base_times) if time > start_time]
    if not indices_after_start_index:
        raise ValueError(f"Scenario start time {start_time} is set after the baseline time range")

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index
