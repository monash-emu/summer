"""
Stochastic solver.
"""
from typing import Optional

import numpy as np
from numba import jit

NO_DESTINATION = -1


def build_flow_map(flows) -> np.ndarray:
    """
    Create an list that maps flows to the source and destination compartments.
    This is done because numba like ndarrays, and numba is fast.
    """
    flow_map = []
    for flow_idx, flow in enumerate(flows):
        if not flow.source:
            # This is used for transition and exit flows.
            # Entry flows and handled separately.
            continue

        flow_map_el = [flow_idx, flow.source.idx, NO_DESTINATION]
        if flow.dest:
            flow_map_el[2] = flow.dest.idx

        flow_map.append(flow_map_el)

    # Convert to a matrix because Numba likes matrices.
    return np.array(flow_map, dtype=np.int)


def sample_transistion_flows(seed, flow_rates, flow_map, comp_vals, timestep):
    """
    Returns a 1D array of changes to each compartment due to transition and exit flows.
    The number of transitions are calculated by sampling from a multinomial distribution.

    The probability of an individual leaving a compartment, and their probability
    of leaving via a given flow is calculated according to the method outlined in this doc:

        https://autumn-files.s3-ap-southeast-2.amazonaws.com/Switching_to_stochastic_mode.pdf

    Args:
        seed: random seed for multinomial sampling
        flow_rates: 2D FxC array (F - flows, C - compartments) of flow rates out of a given compartment
        flow_map: 2D array mapping of flow idxs to compartment idxs, creating using `build_flow_map`.
        comp_vals: 1D array of current compartment values
        timestep: timestep being used for solver

    """
    # Normalize flow rates by compartment size
    flow_rates_normalized = np.true_divide(
        flow_rates,
        comp_vals,
        np.zeros_like(flow_rates),
        where=comp_vals != 0,
    )

    # Find sum of normalized flow weights per compartment
    total_flows = flow_rates_normalized.sum(axis=0)

    # Find the proportion of people who leave a given compartment via a given flow
    prop_flow = np.true_divide(
        flow_rates_normalized,
        total_flows,
        np.zeros_like(flow_rates_normalized),
        where=total_flows != 0,
    )

    # Find probability of a single person leaving a given compartment
    p_stay = np.exp(-1 * total_flows * timestep)
    p_leave = 1 - p_stay

    # Find the probability that a person leaves via a given flow (add stay probability as final row)
    # This is a (F + 1) x C matrix where
    #  - F is the number of flows
    #  - C is the number of compartments
    #  - each element is a probability of a person leaving via a given flow
    flow_probs = np.vstack((p_leave * prop_flow, p_stay))

    # Sample the transition and exit flows for each compartment using a multinomial.
    # So that we know how many people left the compartment for each flow.
    sampled_flows = np.zeros(flow_probs.shape)
    _sample_flows_multinomial(seed, sampled_flows, comp_vals, flow_probs)

    # Map exit flows to their destinations and subtract exit flows from the sources.
    # This will give us an array of changes in compartment sizes.
    transition_changes = np.zeros_like(comp_vals)
    if flow_map.size > 0:
        _map_sampled_flows(transition_changes, flow_map, sampled_flows)

    return transition_changes


@jit(nopython=True)
def _sample_flows_multinomial(
    seed: Optional[int],
    sampled_flows: np.ndarray,
    comp_vals: np.ndarray,
    flow_probs: np.ndarray,
) -> np.ndarray:
    """
    Sample the transition and exit flows for each compartment using a multinomial.
    So that we know how many people left the compartment for each flow.

    The JIT saves ~5s of runtime.
    """
    if seed is not None:
        np.random.seed(seed)

    for c_idx in range(comp_vals.shape[0]):
        comp_flow_probs = flow_probs[:, c_idx]
        comp_size = comp_vals[c_idx]
        sampled_flows[:, c_idx] = np.random.multinomial(comp_size, comp_flow_probs)


@jit(nopython=True)
def _map_sampled_flows(comp_changes: np.ndarray, flow_map: np.ndarray, sampled_flows: np.ndarray):
    """
    Map exit flows to their destinations and subtract exit flows from the sources.
    This will give us an array of changes in compartment sizes.

    The JIT saves ~1s of runtime.
    """
    for i in range(flow_map.shape[0]):
        f_idx, src_idx, dest_idx = flow_map[i]
        # Source flow (there will always be a source, entry flows handled elsewhere)
        flow_amount = sampled_flows[f_idx, src_idx]
        # Source compartment loses people.
        comp_changes[src_idx] -= flow_amount

        # Dest flow (when it is a transition, rather than an exit)
        if dest_idx > NO_DESTINATION:
            # Destination compartment gains people.
            comp_changes[dest_idx] += flow_amount


def sample_entry_flows(seed: Optional[int], entry_flow_rates: np.ndarray, timestep: float):
    """
    Returns a 1D array of new arrivals to each compartment due to entry flows.
    The new arrivals are calculated by sampling from a Poisson distribution, where
    the provided entry flow rates are assumed to be the mean number of arrivals.

    https://en.wikipedia.org/wiki/Poisson_distribution

    Args:
        seed: random seed for Poisson sampling
        entry_flow_rates: 1D array of mean net entry flows per time-unit into each compartment
        timestep: timestep being used for solver

    """
    entry_changes = np.zeros_like(entry_flow_rates)
    _sample_entry_flows(seed, entry_changes, entry_flow_rates, timestep)
    return entry_changes


@jit(nopython=True)
def _sample_entry_flows(
    seed: Optional[int], entry_changes: np.ndarray, entry_flow_rates: np.ndarray, timestep: float
):
    """
    Figure out changes in compartment sizes due to entry flows.
    Sample entry flows using Poisson distribution

    The JIT saves ~0.2s of runtime... maybe.
    """
    if seed is not None:
        np.random.seed(seed)

    # Lambda is the expected value of the Poisson distribution for each compartment.
    lambdas = entry_flow_rates * timestep
    for l_idx in range(lambdas.shape[0]):
        entry_changes[l_idx] = np.random.poisson(lambdas[l_idx])
