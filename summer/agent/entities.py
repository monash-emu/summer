from abc import ABC

from .fields import BaseField, NetworkField


class EntityMetaclass(type):
    pass


class BaseEntity(metaclass=EntityMetaclass):
    @classmethod
    def get_fields(cls):
        """
        Returns and iterator of all the fields.
        """
        for name, field in cls.__dict__.items():
            if issubclass(field.__class__, BaseField):
                yield name, field

    def __str__(self):
        return f"<{self.__class__.__name__}>"


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
