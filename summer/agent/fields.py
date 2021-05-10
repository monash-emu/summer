from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx


class BaseField(ABC):
    """
    A data field representing a component, which can be attached to an entity.
    Eg. the age (component/field) of a person (entity).
    """

    @abstractmethod
    def setup(self, expected_number: int):
        """
        Create the initial data store for that field.
        """
        pass

    @abstractmethod
    def validate(self, value):
        """
        Asserts that the value is of the correct type for the field.
        """
        pass


class IntegerField(BaseField):
    def __init__(self, default: int = None, distribution: Callable[[], int] = None):
        assert (
            distribution or default
        ), "Must specify a default or a distribution for an IntegerField."
        assert not (
            distribution and default
        ), "Cannot specify both a default and a distribution for an IntegerField."
        validate_value = distribution() if distribution else default
        self.validate(validate_value)
        self.default = default
        self.distribution = distribution

    def validate(self, value):
        assert type(value) in (int, np.int64), "IntegerField values must be of int type."

    def setup(self, expected_number: int):
        if self.default:
            arr = np.ones(expected_number, dtype=np.int)
            arr *= self.default
        else:
            arr = np.zeros(expected_number, dtype=np.int)
            for i in range(expected_number):
                arr[i] = self.distribution()

        return arr


class NetworkField(BaseField):
    def setup(self, expected_number: int):
        arr = np.empty(expected_number, dtype=np.object)
        for i in range(expected_number):
            arr[i] = nx.Graph()

        return arr

    def validate(self, value):
        assert type(value) is nx.Graph, "NetworkField values must be a Networkx Graph."