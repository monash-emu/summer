import random
from typing import List, Optional

import numpy as np

from .entities import BaseEntity, BaseNetwork


class Registry:
    """
    Holds data about an entity type and is used to query that data.
    """

    def __init__(self, cls: BaseEntity, initial_number: int = 0):
        self.cls = cls
        self.vals = {}
        self._is_network = issubclass(cls, BaseNetwork)
        self._initial_number = initial_number
        self.reset()

    def reset(self):
        """
        Reset all fields
        """
        self.count = self._initial_number
        for field_name, field in self.cls.fields.items():
            self.vals[field_name] = field.setup(self._initial_number)

    @property
    def query(self):
        return Query(self)

    def add(self, entity: BaseEntity, id: Optional[int] = None) -> int:
        assert getattr(entity, "id", None) is None, "Cannot create an entity with an existing ID."
        entity_id = self.count
        entity.id = entity_id
        self.count += 1
        for field_name in self.cls.fields.keys():
            # Add each field to its value array.
            val = getattr(entity, field_name)
            is_obj_arr = self.vals[field_name].dtype == np.dtype("O")
            if is_obj_arr:
                # This is really dumb but you can't construct a NumPy object array with
                # only empty nx.Graph() objects - possibly because they're falsey,
                # so I put in this True value then immediately throw it away.
                # Please fix this if you can.
                els = self.vals[field_name].tolist() + [val, True]
                self.vals[field_name] = np.array(els, dtype=object)[:-1].copy()

            else:
                self.vals[field_name] = np.append(self.vals[field_name], [val])

        return entity_id

    def get(self, entity_id: int) -> BaseEntity:
        """
        Returns an instance of the entity's class, populated with the data form its compoenent.
        """
        assert entity_id < self.count, f"Entity ID {entity_id} out of range"
        entity_values = {k: self.vals[k][entity_id] for k in self.cls.fields.keys()}
        entity = self.cls(**entity_values)
        entity.id = entity_id
        return entity

    def write(self, entity_id: int, field_name: str, value) -> BaseEntity:
        """
        Sets a value on a particular entity.
        """
        assert entity_id < self.count, f"Entity ID {entity_id} out of range"
        self.vals[field_name][entity_id] = value

    def save(self, enitites: List[BaseEntity], field_names=None):
        """
        Save a list of entities to the registry.
        """
        fields_to_update = field_names if field_names is not None else self.cls.fields.keys()
        for entity in enitites:
            for field_name in fields_to_update:
                self.vals[field_name][entity.id] = getattr(entity, field_name)

    def add_node(self, entity_id: int, node_id: int):
        """
        Network only method.
        Add a node to the network's graph.
        Assumes the network is fully connected.
        """
        assert self._is_network, "This method can only be used for Network entities"
        assert entity_id < self.count, f"Entity ID {entity_id} out of range"
        # Add the node to the graph
        graph = self.vals["graph"][entity_id]
        assert not node_id in graph, f"Agent {node_id} already in {self.cls.__name__} {entity_id}"
        graph.add_node(node_id)
        edges = []
        for dest_id in graph.nodes:
            if dest_id != node_id:
                edges.append((node_id, dest_id))

        graph.add_edges_from(edges)

        # Update the size
        self.vals["size"][entity_id] += 1

    def remove_node(self, entity_id: int, node_id: int):
        """
        Network only method.
        Remove an node from the graph.
        Assumes the network is fully connected.
        """
        assert self._is_network, "This method can only be used for Network entities"
        assert entity_id < self.count, f"Entity ID {entity_id} out of range"
        graph = self.vals["graph"][entity_id]
        assert node_id in graph, f"Node {node_id} not in {self.cls} {entity_id}"
        graph.remove_node(node_id)

    def get_nodes(self, entity_id: int) -> np.ndarray:
        """
        Returns the node IDs of a  network.
        """
        assert self._is_network, "This method can only be used for Network entities"
        assert entity_id < self.count, f"Entity ID {entity_id} out of range"
        graph = self.vals["graph"][entity_id]
        return np.array(list(graph.nodes.keys()), dtype=np.int64)

    def get_node_contacts(self, entity_id: int, node_id: int):
        """
        Returns the direct contacts/network neighbours of a node.
        """
        assert self._is_network, "This method can only be used for Network entities"
        assert entity_id < self.count, f"Entity ID {entity_id} out of range"
        graph = self.vals["graph"][entity_id]
        return list(graph.neighbors(node_id))


class Query:
    FILTER_STRATEGIES = (
        # Greater than
        ("__gt", lambda a, b: a > b),
        # Less than
        ("__lt", lambda a, b: a < b),
        # Greater than or equal to
        ("__gte", lambda a, b: a >= b),
        # Less than or equal to
        ("__lte", lambda a, b: a <= b),
        # Not equal
        ("__ne", lambda a, b: a != b),
        # Equal
        ("", lambda a, b: a == b),
    )

    def __init__(self, registry: Registry, ids: Optional[np.array] = None):
        self._registry = registry
        if ids is None:
            self._ids = np.array(range(self._registry.count), dtype=np.int64)
        elif type(ids) is np.ndarray and ids.dtype is np.int64:
            self._ids = ids
        else:
            self._ids = np.array(ids, dtype=np.int64)

    def filter(self, **kwargs):
        """
        Returns a Query with entities filtered according to provided arguments.

        A filter kwarg ending with one of the following suffixes with use a different comparison strategy.
        Strategies:
            - __gt:  Greater than
            - __lt:  Less than
            - __gte: Greater than equal
            - __lte: Less than equal
            - __ne:  Not equal

        Example: "select all entities with an age greater than 12 and weight equal to 1"

        query.filter(age__gt=12, weight=1)

        """
        filtered_ids = self._filter(**kwargs)
        new_ids = np.intersect1d(self._ids, filtered_ids)
        return Query(self._registry, new_ids)

    def exclude(self, **kwargs):
        """
        Opposite of filter - throw away matching ids.
        """
        filtered_ids = self._filter(**kwargs)
        new_ids = np.setdiff1d(self._ids, filtered_ids)
        return Query(self._registry, new_ids)

    def _filter(self, **kwargs):
        mask = None
        for condition, value in kwargs.items():
            m = None
            for suffix, strategy in Query.FILTER_STRATEGIES:
                if condition.endswith(suffix):
                    c = condition[: (-1 * len(suffix))] if suffix else condition
                    self._registry.cls.assert_fieldname(c)
                    m = strategy(self._registry.vals[c], value)
                    break

            mask = m if mask is None else (mask * m)

        # Filter IDs based on mask
        return np.where(mask)[0]

    def choose(self, num_choices: int):
        """
        Returns a Query with entities filtered down to `num_choices` randomly chosen entities.
        """
        new_ids = random.choices(self._ids, k=num_choices)
        return Query(self._registry, new_ids)

    def select(self, entity_ids: List[int]):
        """
        Returns a Query with entities filtered down to those provided as `entity_ids`.
        """
        new_ids = np.intersect1d(self._ids, entity_ids)
        return Query(self._registry, new_ids)

    def deselect(self, entity_ids: List[int]):
        """
        Returns a Query with entities filtered down to those NOT provided as `entity_ids`.
        """
        new_ids = np.setdiff1d(self._ids, entity_ids)
        return Query(self._registry, new_ids)

    def where(self, func):
        """
        Returns the agent ids where the provided func evalutates to True.
        Function may be vectorized or request entity specific arguments.
        """
        if self._is_vectorized(func):
            # Get the mask with the vectorized function.
            args = self._get_requested_args(func)
            mask = func(*args)
        else:
            # Get the mask with the scalar function.
            mask = np.empty(len(self._ids), dtype=np.bool)
            for idx, entity_id in enumerate(self._ids):
                args = self._get_requested_args(func, entity_id=entity_id)
                mask[idx] = func(*args)

        new_ids = self._ids[mask]
        return Query(self._registry, new_ids)

    def update(self, **kwargs):
        """
        Update all selected entries with provided data.
        Function may be vectorized or request entity specific arguments.
        """
        for field_name, value in kwargs.items():
            self._registry.cls.assert_fieldname(field_name)
            if callable(value) and self._is_vectorized(value):
                # Update the fields according to the vectorized function.
                args = self._get_requested_args(value)
                update_values = value(*args)
            elif callable(value):
                # Update the fields according to the scalar function.
                update_values = np.empty(
                    len(self._ids), dtype=self._registry.vals[field_name].dtype
                )
                for idx, entity_id in enumerate(self._ids):
                    args = self._get_requested_args(value, entity_id=entity_id)
                    update_values[idx] = value(*args)
            else:
                # Just set everything to the single, static value.
                update_values = value

            self._registry.vals[field_name][self._ids] = update_values

    def all(self) -> List[BaseEntity]:
        """
        Returns all entities selected by the query as class instances.
        """
        entities = []
        for entity_id in self._ids:
            entities.append(self._registry.get(entity_id))

        return entities

    def only(self) -> BaseEntity:
        """
        Returns the class instance of the only remaining selected entity.
        """
        assert len(self._ids) == 1, "Can only use `only()` when one id is selected."
        return self._registry.get(self._ids[0])

    def id(self) -> List[int]:
        """
        Returns the ID of the only remaining selected entity.
        """
        assert len(self._ids) == 1, "Can only use `id()` when one id is selected."
        return self._ids[0]

    def ids(self) -> List[int]:
        return list(self._ids)

    def _get_requested_args(self, func, entity_id: Optional[int] = None):
        """
        Returns the arguments requested using the `@arguments` decorator.
        """
        arg_names = getattr(func, "_arg_names", [])
        return [self._get_requested_arg(n, entity_id) for n in arg_names]

    def _get_requested_arg(self, field_name, entity_id: Optional[int]):
        if entity_id is None:
            # Return vectorized argument for all selected ids.
            if field_name == "id":
                return self._ids
            else:
                return self._registry.vals[field_name][self._ids]
        else:
            # Return scalar argument for the given id.
            if field_name == "id":
                return entity_id
            else:
                return self._registry.vals[field_name][entity_id]

    def _is_vectorized(self, func):
        """
        Returns True if the given function is marked using @vectorized,
        which means it can accept array arguments and returns arrays.
        """
        return bool(getattr(func, "_vectorized", None))


def vectorized(func):
    """
    Decorator which marks an update function as vectorized.
    """
    func._vectorized = True
    return func


def arguments(*arg_names):
    """
    Decorator which defines the arguments for an update function.
    """

    def decorator(func):
        func._arg_names = arg_names
        return func

    return decorator
