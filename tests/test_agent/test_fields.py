import pytest

import numpy as np
import networkx as nx
from numpy.testing import assert_array_equal
from summer.agent.fields import IntegerField, NetworkField


def test_integer_field_init_validation():
    # No default or distribution
    with pytest.raises(AssertionError):
        IntegerField()

    # Default not an int
    with pytest.raises(AssertionError):
        IntegerField(default=1.23)

    # Default is an int
    IntegerField(default=1)

    # Distribution result not an int
    with pytest.raises(AssertionError):
        IntegerField(distribution=lambda: 1.23)

    # Distribution result is an int
    IntegerField(distribution=lambda: 1)


def test_integer_field_setup__with_default():
    f = IntegerField(default=1)
    arr = f.setup(expected_number=0)
    assert_array_equal(arr, np.array([], dtype=np.int))

    arr = f.setup(expected_number=3)
    assert_array_equal(arr, np.array([1, 1, 1], dtype=np.int))


def test_integer_field_setup__with_distribution():
    class Dist:
        def __init__(self):
            self.count = 0

        def __call__(self):
            self.count += 1
            return self.count

    f = IntegerField(distribution=Dist())
    arr = f.setup(expected_number=0)
    assert_array_equal(arr, np.array([], dtype=np.int))

    arr = f.setup(expected_number=3)
    # Gets called once in field's __init__ method.
    assert_array_equal(arr, np.array([2, 3, 4], dtype=np.int))


def test_network_field_setup():
    f = NetworkField()
    arr = f.setup(expected_number=0)
    assert_array_equal(arr, np.array([], dtype=object))

    arr = f.setup(expected_number=3)
    assert arr.size == 3
    assert all(type(g) is nx.Graph for g in arr)
    assert len(set(id(g) for g in arr)) == 3
    assert all(g.number_of_nodes() == 0 for g in arr)
