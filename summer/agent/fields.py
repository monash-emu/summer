from typing import Callable, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx


class BaseField(ABC):
    """
    A data field representing a component, which can be attached to an entity.
    Eg. the age (component/field) of a person (entity).
    """

    default: Optional[Any]
    distribution: Optional[Callable[[], Any]]

    @abstractmethod
    def __init__(self):
        self.default = None
        self.distribution = None

    @abstractmethod
    def setup(self, initial_number: int):
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
        has_dist = distribution is not None
        has_default = default is not None
        assert (
            has_dist or has_default
        ), "Must specify a default or a distribution for an IntegerField."
        assert not (
            has_dist and has_default
        ), "Cannot specify both a default and a distribution for an IntegerField."
        validate_value = distribution() if has_dist else default
        self.validate(validate_value)
        self.default = default
        self.distribution = distribution

    def validate(self, value):
        assert type(value) in (int, np.int64), "IntegerField values must be of int type."

    def setup(self, initial_number: int):
        if self.default:
            arr = np.ones(initial_number, dtype=np.int)
            arr *= self.default
        else:
            arr = np.zeros(initial_number, dtype=np.int)
            for i in range(initial_number):
                arr[i] = self.distribution()

        return arr


class GraphField(BaseField):
    """
    Represents an undirected graph using a NetworkX Graph object.
    """

    def __init__(self):
        super().__init__()
        self.distribution = self._get_default

    def _get_default(self):
        return nx.Graph()

    def setup(self, initial_number: int):
        arr = np.empty(initial_number, dtype=np.object)
        for i in range(initial_number):
            arr[i] = self._get_default()

        return arr

    def validate(self, value):
        assert type(value) is nx.Graph, "GraphField values must be a Networkx Graph."