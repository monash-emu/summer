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

agent = Agent(
    recovery_date=-1,
    disease=Disease.SUSCEPTIBLE
)

   


#1 save
agent = querything.get_agent(1)
agent.disease = 2
agent.save()

#2 override setters n getters
agent = querything.get_agent(1)
agent.disease = 2

#3 post operation bulk update thingy
agent = querything.get_agent(1)
agent.disease = 2
querything.update_all([agent])

# All agents due to recover today should recover
model.agents.filter(recovery_date=time).update(disease=3)





# Agent proxy
agent = Agent(
    recovery_date=-1,
    disease=Disease.SUSCEPTIBLE
)
agent = agent.rsqdqwdqwd
agent.id




class TribeNetwork(BaseNetwork):
    capacity = fields.IntegerField(distribution=np.random.poisson)


model = AgentModel(start_time=0, end_time=10, timestep=1)
model.set_agent_class(Agent, expected_number=NUM_PEOPLE)
model.set_network_class(TribeNetwork)


@model.setup_step
def setup_tribes(model: AgentModel):
    for _ in range(NUM_PEOPLE):
        agent = Agent(model.next_agent_id, Disease.SUSCEPTIBLE)
        model.add_agent(agent)
        for network in model._networks:
            if network.size < network.capacity:
                network.add_agent(agent.id)
                break
        else:
            network = TribeNetwork(model.next_network_id)
            network.add_agent(agent.id)
            model.add_network(network)


    @summer.array_friendly
    def myfunc():
        sqswq

    # Randomly set three people to infected
    model.agents
        .choose_random(3)
        .update(disease=Disease.INFECTED, recovery_date=myfunc)

    # deep in backend
    if just_a_reg_function:
        for
    elif myfunc.is_parallel:



    inf_agents = random.choices(model._agents, k=3)
    for agent in inf_agents:
        agent.disease = Disease.INFECTED
        agent.recovery_date = round(random.random() * 10)


@model.runtime_step
def recovery_system(model: AgentModel, time: float):
    for agent in model._agents:
        if agent.disease == 2 and agent.recovery_date <= time:
            print(f"Agent {agent.id} recovered at time {time}.")
            agent.recovery_date = None
            agent.disease = 3  # Recovered


@model.runtime_step
def infection_system(model: AgentModel, time: float):
    for network in model._networks:
        for agent_id in network.agents:
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
