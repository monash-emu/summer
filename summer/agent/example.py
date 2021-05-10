import random

import numpy as np

from . import fields
from .entities import BaseAgent, BaseNetwork
from .model import AgentModel

NUM_PEOPLE = 20


class Disease:
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3


class Agent(BaseAgent):
    recovery_date = fields.IntegerField(default=-1)
    disease = fields.IntegerField(default=Disease.SUSCEPTIBLE)


class TribeNetwork(BaseNetwork):
    capacity = fields.IntegerField(distribution=np.random.poisson)


model = AgentModel(start_time=0, end_time=10, timestep=1)
model.set_agent_class(Agent, expected_number=NUM_PEOPLE)
model.set_network_class(TribeNetwork)


@model.setup_step
def setup_tribes(model: AgentModel):
    # Create some tribe networks and assign each agent to a tribe.
    for agent_id in model.agents.ids():
        for network in model._networks:
            if network.size < network.capacity:
                network.add_agent(agent_id)
                break
        else:
            network = TribeNetwork()
            network.add_agent(agent_id)
            model.networks.add(network)

    # Randomly set three people to infected
    get_recovery_date = lambda: round(random.random() * 10)
    model.agents.query.choose(3).update(disease=Disease.INFECTED, recovery_date=get_recovery_date)


@model.runtime_step
def recovery_system(model: AgentModel, time: float):
    model.agents.query.filter(disease=Disease.INFECTED, recovery_date__lte=time).update(
        disease=Disease.RECOVERED
    )


@model.runtime_step
def infection_system(model: AgentModel, time: float):
    for network in model.networks.query.entities():
        for agent_id in network.agent_ids:
            agent = model.get_agent(agent_id)
            if not agent.disease == 2:  # Infected
                continue

            contact_ids = network.get_contacts(agent_id)
            for contact_id in contact_ids:
                contact = model.get_agent(contact_id)
                if not contact.disease == 1:  # Susceptible
                    continue

                transmission_pr = 0.3
                if random.random() <= transmission_pr:
                    contact.disease = 2
                    contact.recovery_date = time + round(random.random() * 10)
                    print(f"Agent {agent_id} infected agent {contact_id} in {network}")
