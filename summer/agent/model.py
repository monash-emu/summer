import numpy as np

from .agent import BaseAgent
from .network import BaseNetwork


class AgentModel:
    def __init__(self, start_time: float, end_time: float, timestep: float):
        # Times
        assert end_time > start_time, "End time must be greater than start time"
        time_period = end_time - start_time
        num_steps = 1 + (time_period / timestep)
        msg = f"Time step {timestep} must be less than time period {time_period}"
        assert num_steps >= 1, msg
        msg = f"Time step {timestep} must be a factor of time period {time_period}"
        assert num_steps % 1 == 0, msg
        self._times = np.linspace(start_time, end_time, num=int(num_steps))
        self._timestep = timestep

        self._networks = []
        self._agents = []

    @property
    def next_network_id(self):
        return len(self._agents)

    @property
    def next_agent_id(self):
        return len(self._networks)

    def add_agent(self, agent: BaseAgent):
        self._agents.append(agent)

    def add_network(self, network: BaseNetwork):
        self._networks.append(network)

    def run(self):
        # Do setup stuff
        for time in self._times:
            self._run_timestep(time)

    def _run_timestep(self, time):
        # Infection
        for network in self._networks:
            for agent_id in network.agents:
                contact_ids = network.get_contacts(agent_id)
                for contact_id in contact_ids:
                    print(f"Agent {agent_id} contacted agent {contact_id} in {network}")
