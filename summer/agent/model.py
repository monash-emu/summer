import numpy as np
import random

from .entities import BaseAgent, BaseNetwork
from .registry import Registry


class AgentModel:
    agents: Registry
    networks: Registry

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

        self._setup_systems = []
        self._runtime_systems = []

        self.agents = None
        self.networks = None

        self._post_setup_hook = None
        self._post_timestep_hook = None

    def add_setup_step(self, func):
        self._setup_systems.append(func)

    def add_runtime_step(self, func):
        self._runtime_systems.append(func)

    def set_agent_class(self, agent_class: BaseAgent, initial_number: int = 0):
        self.agents = Registry(agent_class, initial_number)

    def set_network_class(self, network_class: BaseNetwork, initial_number: int = 0):
        self.networks = Registry(network_class, initial_number)

    def add_setup_hook(self, hook):
        """
        Register a function which will be run on the model just after setup
        """
        self._post_setup_hook = hook

    def add_timestep_hook(self, hook):
        """
        Register a function which will be run on the model after each timestep
        """
        self._post_timestep_hook = hook

    def run(self):
        assert self.agents, "An agent class must be set before running the model."
        assert self.networks, "A network class must be set before running the model."
        self._reset()
        self._run_setup()
        if self._post_setup_hook:
            self._post_setup_hook(self)

        for time in self._times:
            self._run_timestep(time)
            if self._post_timestep_hook:
                self._post_timestep_hook(self, time)

    def _reset(self):
        self.agents.reset()
        self.networks.reset()

    def _run_setup(self):
        for setup_step in self._setup_systems:
            setup_step(self)

    def _run_timestep(self, time):
        for runtime_step in self._runtime_systems:
            runtime_step(self, time)
