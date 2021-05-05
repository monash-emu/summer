from typing import List, Callable, Any, Set, Dict
import numpy as np
import random

from .entities import BaseAgent, BaseNetwork
from .query import Query


class AgentModel:
    def __init__(
        self,
        start_time: float,
        end_time: float,
        timestep: float,
    ):
        # Setup run timesteps
        assert end_time > start_time, "End time must be greater than start time"
        time_period = end_time - start_time
        num_steps = 1 + (time_period / timestep)
        msg = f"Time step {timestep} must be less than time period {time_period}"
        assert num_steps >= 1, msg
        msg = f"Time step {timestep} must be a factor of time period {time_period}"
        assert num_steps % 1 == 0, msg
        self._times = np.linspace(start_time, end_time, num=int(num_steps))
        self._timestep = timestep

        # Setup initial networks
        self._network_cls = None
        self._network_vals = {}
        self._network_max_id = 0

        # Setup agents
        self._agent_cls = None
        self._agent_vals = {}
        self._agent_max_id = 0

        self._setup_systems = []
        self._runtime_systems = []

    def setup_step(self):
        def decorator(f):
            self._setup_systems.append(f)
            return f

        return decorator

    def runtime_step(self):
        def decorator(f):
            self._systems.append(f)
            return f

        return decorator

    @property
    def agents(self):
        return Query(self, self._agent_field_names, self._agent_vals, self._agent_max_id)

    @property
    def networks(self):
        return Query(self, self._network_field_names, self._network_vals, self._network_max_id)

    def set_agent_class(self, agent_class: BaseAgent, expected_number: int = 0):
        self._agent_cls = agent_class
        self._agent_count = expected_number
        for name, field in agent_class.fields:
            self._agent_field_names.add(name)
            self._agent_vals[name] = field.setup(expected_number)

    def set_network_class(self, network_class: BaseNetwork, expected_number: int = 0):
        self._network_cls = network_class
        self._network_count = expected_number
        for name, field in network_class.fields:
            self._network_field_names.add(name)
            self._network_vals[name] = field.setup(expected_number)

    def run(self):
        assert self._agent_cls, "An agent class must be set before running the model."
        assert self._network_cls, "A network class must be set before running the model."
        self._run_setup()
        for time in self._times:
            self._run_timestep(time)

    def _run_setup(self):
        for setup_step in self._setup_systems:
            setup_step(self)

    def _run_timestep(self, time):
        for setup_step in self._setup_systems:
            setup_step(self)

        # Run recoveries
        # TODO: Turn into a system
        for agent in self._agents:
            if agent.disease == 2 and agent.recovery_date <= time:
                print(f"Agent {agent.id} recovered at time {time}.")
                agent.recovery_date = None
                agent.disease = 3  # Recovered

        # Run infections
        # TODO: Turn into a system
        for network in self._networks:
            for agent_id in network.agents:
                agent = self.get_agent(agent_id)
                if not agent.disease == 2:  # Infected
                    continue

                contact_ids = network.get_contacts(agent_id)
                for contact_id in contact_ids:
                    contact = self.get_agent(contact_id)
                    if not contact.disease == 1:  # Susceptible
                        continue

                    transmission_pr = 0.3
                    if random.random() <= transmission_pr:
                        contact.disease = 2
                        contact.recovery_date = time + round(random.random() * 10)
                        print(f"Agent {agent_id} infected agent {contact_id} in {network}")
