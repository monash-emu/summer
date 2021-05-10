from abc import ABC
from typing import Dict

from .fields import BaseField, NetworkField


class EntityMeta(type):
    def __new__(cls, name, bases, dct):
        """
        Annotate new Entity class with 'fields' attributes.
        """
        new_cls = super().__new__(cls, name, bases, dct)
        fields = {}
        for name, field in dct.items():
            if issubclass(field.__class__, BaseField):
                fields[name] = field

        new_cls.fields = fields
        return new_cls


class BaseEntity(metaclass=EntityMeta):
    fields: Dict[str, BaseField]

    def __init__(self, **kwargs):
        for field_name, field in self.fields.items():
            field_val = kwargs.get(field_name)
            if not field_val and field.default:
                field_val = field.default
            elif not field_val and field.distribution:
                field_val = field.distribution()
            elif not field_val:
                raise ValueError(f"Field {field_name} does not have a value.")

            assert field_val is not None
            field.validate(field_val)
            setattr(self, field_name, field_val)

    @classmethod
    def assert_fieldname(cls, field_name):
        assert (
            field_name in cls.fields.keys()
        ), f"Could not find field {field_name} in entity {cls}."


class BaseAgent(BaseEntity):
    pass


class BaseNetwork(BaseEntity):
    graph = NetworkField()

    @property
    def agents_ids(self):
        return self.graph.nodes

    def add_agent(self, agent_id: int):
        """
        Add an agent to the graph, with fully connected edges.
        """
        assert not agent_id in self.graph, f"Agent {agent_id} already in {self}"
        msg = f"Cannot add agent {agent_id} to {self}: it is full."
        assert self.has_capacity, msg
        self.graph.add_node(agent_id)
        edges = []
        for dest_id in self.graph.nodes:
            if dest_id != agent_id:
                edges.append((agent_id, dest_id))

        self.graph.add_edges_from(edges)

    def remove_agent(self, agent_id: int):
        """
        Remove an agent from the graph.
        """
        assert agent_id in self.graph, f"Agent {agent_id} not in network {self.id}"
        self.graph.remove_node(agent_id)

    @property
    def size(self):
        return self.graph.number_of_nodes()

    def get_contacts(self, agent_id: int):
        return self.graph.neighbors(agent_id)
