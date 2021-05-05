from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx


class BaseField(ABC):
    pass
    # @abstractmethod
    # def clean(self, value):
    #     pass

    # @abstractmethod
    # def validate(self, value):
    #     pass

    @abstractmethod
    def setup(self, expected_number: int):
        pass


class IntegerField(BaseField):
    def __init__(self, default: int = None, distribution: Callable[[], int] = None):
        assert (
            distribution or default
        ), "Must specify a default or a distribution for an IntegerField."
        assert not (
            distribution and default
        ), "Cannot specify both a default and a distribution for an IntegerField."
        if distribution:
            assert type(distribution()) is int, "Distribution value for IntegerField must be an int"
        else:
            assert type(default) is int, "Default value for IntegerField must be an int"

        self.default = default
        self.distribution = distribution

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
