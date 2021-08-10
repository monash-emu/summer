"""
Optimized 'hot' functions used by CompartmentalModel and its runners.
"""
from abc import ABC, abstractmethod

import numba
import numpy as np

class ComputedValueProcessor(ABC):
    """
    Base class for computing (runtime) derived values
    """
    def __init__(self):
        pass

    def prepare_to_run(self, compartments, flows):
        """Doing any pre-computation or setup that requires information about model structure

        Args:
            compartments ([type]): [description]
            flows ([type]): [description]
        """
        pass

    @abstractmethod
    def process(self, compartment_values, computed_values, time):
        pass

# Use Numba to speed up the calculation of the population.
@numba.jit(nopython=True)
def find_sum(compartment_values: np.ndarray) -> float:
    return compartment_values.sum()

@numba.jit(nopython=True)
def accumulate_positive_flow_contributions(
    flow_rates: np.ndarray,
    comp_rates: np.ndarray,
    pos_flow_map: np.ndarray,
):
    """
    Fast accumulator for summing positive flow rates into their effects on compartments

    Args:
        flow_rates (np.ndarray): Flow rates to be accumulated
        comp_rates (np.ndarray): Output array of compartment rates
        pos_flow_map (np.ndarray): Array of src (flow), target (compartment) indices
    """
    for src, target in pos_flow_map:
        comp_rates[target] += flow_rates[src]


@numba.jit(nopython=True)
def accumulate_negative_flow_contributions(
    flow_rates: np.ndarray,
    comp_rates: np.ndarray,
    neg_flow_map: np.ndarray,
):
    """Fast accumulator for summing negative flow rates into their effects on compartments

    Args:
        flow_rates (np.ndarray): Flow rates to be accumulated
        comp_rates (np.ndarray): Output array of compartment rates
        neg_flow_map (np.ndarray): Array of src (flow), target (compartment) indices
    """
    for src, target in neg_flow_map:
        comp_rates[target] -= flow_rates[src]


@numba.jit(nopython=True)
def sparse_pairs_accum(
    map_idx: np.ndarray, compartment_vals: np.ndarray, target_size: int
) -> np.ndarray:
    """Fast equivalent of matrix multiplication AxB, where A is a (sparse) binary matrix (category map),
       and B is compartment_vals

    Args:
        map_idx (np.ndarray): Integer array of size (n_compartments, 2), where each row is a pair of
                              (source (compartment index), target (<target_size))
        compartment_vals (np.ndarray): Values to be selected for accumulation into target
        target_size (int): Number of target categories, and size of output array

    Returns:
        np.ndarray: Array of size target_size, with the accumulated values
    """
    out_arr = np.zeros(target_size)
    for src, target in map_idx:
        out_arr[target] += compartment_vals[src]
    return out_arr


def binary_matrix_to_sparse_pairs(category_matrix: np.ndarray) -> np.ndarray:
    """Converts (sparse) binary matrices into index arrays for use by sparse_pairs_accum

    Args:
        category_matrix (np.ndarray): Binary matrix mapping compartments (row) to categories (column)

    Returns:
        np.ndarray: Integer array of size (n_compartments, 2), where each row is a pair of
                    (source (compartment index), target (category))
    """
    cat_idx = category_matrix.astype(bool)
    out_idx = []
    for i in range(cat_idx.shape[0]):
        for j in range(cat_idx.shape[1]):
            if cat_idx[i, j]:
                out_idx.append((j, i))
    return np.array(out_idx, dtype=int)

