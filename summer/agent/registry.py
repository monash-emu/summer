import random
import numpy as np
from typing import List, Optional
from .entities import BaseEntity


class Registry:
    """
    Holds data about an entity type and is used to query that data.
    """

    def __init__(self, cls: BaseEntity, expected_number: int = 0):
        self.cls = cls
        self.vals = {}
        self.count = expected_number
        for field_name, field in cls.fields.items():
            self.vals[field_name] = field.setup(expected_number)

    @property
    def query(self):
        return Query(self)

    def add(self, entity: BaseEntity, id: Optional[int] = None) -> int:
        assert getattr(entity, "id", None) is None, "Cannot create an entity with an existing ID."
        entity_id = self.count
        entity.id = entity_id
        self.count += 1
        for field_name in self.cls.fields.keys():
            self.vals[field_name] = np.append(self.vals[field_name], [getattr(entity, field_name)])

        return entity_id

    def get(self, entity_id: int) -> BaseEntity:
        """
        Returns an instance of the entity's class, populated with the data form its compoenent.
        """
        assert entity_id <= self.count, f"Entity ID {entity_id} out of range"
        entity_values = {k: self.vals[k][entity_id] for k in self.cls.fields.keys()}
        entity = self.cls(**entity_values)
        entity.id = entity_id
        return entity

    def write(self, entity_id: int, field_name: str, value) -> BaseEntity:
        """
        Sets a value on a particular entity.
        """
        assert entity_id <= self.count, f"Entity ID {entity_id} out of range"
        self.vals[field_name][entity_id] = value

    def save(self, enitites: List[BaseEntity], field_names=None):
        """
        Save a list of entities to the registry.
        """
        fields_to_update = field_names if field_names is not None else self.cls.fields.keys()
        for entity in enitites:
            for field_name in fields_to_update:
                self.vals[field_name][entity.id] = getattr(entity, field_name)


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
        self._ids = ids if ids is not None else np.array(range(self._registry.count))

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
        filtered_ids = np.where(mask)[0]
        new_ids = np.intersect1d(self._ids, filtered_ids)
        return Query(self._registry, new_ids)

    def when(self, func):
        """
        Returns the agent ids where the provided func evalutates to True.
        Function may be vectorized or request entity specific arguments.
        """
        arg_names = getattr(func, "_arg_names", [])
        if getattr(func, "_vectorized", None):
            # Get the mask with the vectorized function.
            args = [self._registry.vals[n][self._ids] for n in arg_names]
            mask = func(*args)
        else:
            # Get the mask with the scalar function.
            mask = np.empty(len(self._ids), dtype=np.bool)
            for idx, entity_id in enumerate(self._ids):
                args = [self._registry.vals[n][entity_id] for n in arg_names]
                mask[idx] = func(*args)

        new_ids = self._ids[mask]
        return Query(self._registry, new_ids)

    def choose(self, num_choices: int):
        """
        Returns a Query with entities filtered down to `num_choices` randomly chosen entities.
        """
        new_ids = random.choices(self._ids, k=num_choices)
        return Query(self._registry, new_ids)

    def update(self, **kwargs):
        """
        Update all selected entries with provided data.
        Function may be vectorized or request entity specific arguments.
        """
        for field_name, value in kwargs.items():
            self._registry.cls.assert_fieldname(field_name)
            arg_names = getattr(value, "_arg_names", [])
            if callable(value) and getattr(value, "_vectorized", None):
                # Update the fields according to the function, which takes
                # its requested input values as vectors.
                args = [self._registry.vals[n][self._ids] for n in arg_names]
                update_values = value(*args)
            elif callable(value):
                # Update the fields according to the function, which takes
                # its requested input values as scalars (in a for loop).
                update_values = np.empty(
                    len(self._ids), dtype=self._registry.vals[field_name].dtype
                )
                for idx, entity_id in enumerate(self._ids):
                    args = [self._registry.vals[n][entity_id] for n in arg_names]
                    update_values[idx] = value(*args)
            else:
                # Just set everything to the static value.
                update_values = value

            self._registry.vals[field_name][self._ids] = update_values

    def entities(self) -> List[BaseEntity]:
        """
        Returns all entities selected by the query as class instances.
        """
        entities = []
        for entity_id in self._ids:
            entities.append(self._registry.get(entity_id))

        return entities

    def ids(self) -> List[int]:
        return list(self._ids)


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
