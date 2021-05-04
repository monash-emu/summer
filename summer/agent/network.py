from typing import List
from abc import ABC

import networkx as nx


class BaseNetwork(ABC):
    def __init__(self, network_id: int, agent_ids: List[int] = []):
        self.id = network_id
        self.graph = nx.Graph()
        for agent_id in agent_ids:
            self.add_agent(agent_id)

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
            edges.append((agent_id, dest_id))

        self.graph.add_edges_from(edges)

    def remove_agent(self, agent_id: int):
        """
        Remove an agent from the graph.
        """
        assert agent_id in self.graph, f"Agent {agent_id} not in network {self.id}"
        self.graph.remove_node(agent_id)

    @property
    def agents(self):
        return self.graph.nodes

    @property
    def size(self):
        return self.graph.number_of_nodes()

    @property
    def has_capacity(self):
        return True

    def get_contacts(self, agent_id: int):
        return self.graph.neighbors(agent_id)

    def __str__(self):
        return f"<BaseNetwork {self.id}>"
