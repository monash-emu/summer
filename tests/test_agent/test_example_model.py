import random

import numpy as np

from summer.agent import fields
from summer.agent.registry import arguments, vectorized
from summer.agent.entities import BaseAgent, BaseNetwork
from summer.agent.model import AgentModel

NUM_PEOPLE = 100
INTIIAL_INFECTED = 10
MEAN_TRIBE_SIZE = 12
NO_DATE = -1
RECOVERY_LIMIT = 5
DEATH_LIMIT = 4
INFECTION_PR = 0.3
RECOVERY_PR = 0.8


class Disease:
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3
    DEAD = 4


class NetworkType:
    TRIBE = 0
    TRADE = 1


def test_example_model__smoke_test_run():
    model = build_example_model(NUM_PEOPLE, INTIIAL_INFECTED)
    model.run()


def build_example_model(
    num_people: int,
    initial_infected: int,
):

    model = AgentModel(start_time=0, end_time=10, timestep=1)
    model.set_agent_class(Agent, initial_number=num_people)
    model.set_network_class(Network)
    model.add_setup_step(setup_tribes)
    model.add_setup_step(get_seed_infection(initial_infected))
    model.add_runtime_step(recovery_system)
    model.add_runtime_step(death_system)
    model.add_runtime_step(infection_system)
    return model


def poisson_with_floor(lam, floor):
    """
    Sample from a Poisson distribution with a minimum allowed value.
    """

    def poisson():
        assert lam > floor
        result = 0
        while result < floor:
            result = np.random.poisson(lam)
        return result

    return poisson


class Agent(BaseAgent):
    """
    An individual who is a part of a tribe.
    """

    death_date = fields.IntegerField(default=NO_DATE)
    recovery_date = fields.IntegerField(default=NO_DATE)
    disease = fields.IntegerField(default=Disease.SUSCEPTIBLE)


class Network(BaseNetwork):
    """
    A tribe of individuals
    """

    type = fields.IntegerField(default=NetworkType.TRIBE)
    capacity = fields.IntegerField(distribution=poisson_with_floor(MEAN_TRIBE_SIZE, 2))


def setup_tribes(model: AgentModel):
    """
    Sets up tribes of people
    """
    # Create some tribe networks and assign each agent to a tribe.
    for agent_id in model.agents.query.ids():
        for network in model.networks.query.all():
            if network.size < network.capacity:
                model.networks.add_node(network.id, agent_id)
                break
        else:
            network_id = model.networks.add(Network())
            model.networks.add_node(network_id, agent_id)

    # Randomly add some inter tribe connections - "trade connections"
    # Each model is has 2 trade networks with 2 other random tribes
    for src_tribe in model.networks.query.all():
        dest_tribes = model.networks.query.choose(2).all()
        for dest_tribe in dest_tribes:
            trade_network = Network(type=NetworkType.TRADE, capacity=2)
            network_id = model.networks.add(trade_network)
            src_tribe_agent_id = random.choice(model.networks.get_nodes(src_tribe.id))
            dest_tribe_agent_id = random.choice(model.networks.get_nodes(dest_tribe.id))
            model.networks.add_node(network_id, src_tribe_agent_id)
            model.networks.add_node(network_id, dest_tribe_agent_id)


def get_seed_infection(initial_infected: int):
    def seed_infection(model: AgentModel):
        """
        Set intially infected individuals.
        """
        model.agents.query.choose(initial_infected).update(
            disease=Disease.INFECTED, recovery_date=build_choose_recovery_date(0)
        )

    return seed_infection


def recovery_system(model: AgentModel, time: float):
    # Everyone who recovers by this date should be set to recovered
    model.agents.query.filter(
        disease=Disease.INFECTED, recovery_date__lte=time, recovery_date__gte=0
    ).update(disease=Disease.RECOVERED)


def death_system(model: AgentModel, time: float):
    # Everyone who dies by this date should be set to dead
    model.agents.query.filter(
        disease=Disease.INFECTED, death_date__lte=time, death_date__gte=0
    ).update(disease=Disease.DEAD)


def infection_system(model: AgentModel, time: float):
    for network_id in model.networks.query.ids():
        # Select infected agents in the network
        network_agent_ids = model.networks.get_nodes(network_id)
        inf_agent_ids = (
            model.agents.query.select(network_agent_ids).filter(disease=Disease.INFECTED).ids()
        )
        for agent_id in inf_agent_ids:
            # Find contacts of an agent and choose some of them to infect.
            contact_ids = model.networks.get_node_contacts(network_id, agent_id)
            inf_contacts_query = (
                model.agents.query.select(contact_ids)
                .filter(disease=Disease.SUSCEPTIBLE)
                .where(choose_infected)
            )
            # Select some infected to recover.
            recovered_query = inf_contacts_query.where(choose_recovered)
            recovered_query.update(
                disease=Disease.INFECTED, recovery_date=build_choose_recovery_date(time)
            )
            # The remainder die.
            inf_contacts_query.deselect(recovered_query.ids()).update(
                disease=Disease.INFECTED, death_date=build_choose_death_date(time)
            )


def build_choose_recovery_date(time):
    @vectorized
    @arguments("id")
    def choose_recovery_date(ids):
        """
        A person randomly recovers 0-3 days after they get infected.
        """
        return time + np.round(np.random.random(len(ids)) * RECOVERY_LIMIT)

    return choose_recovery_date


def build_choose_death_date(time):
    @vectorized
    @arguments("id")
    def choose_death_date(ids):
        """
        A person randomly dies 0-2 days after they get infected.
        """
        return time + np.round(np.random.random(len(ids)) * DEATH_LIMIT)

    return choose_death_date


@vectorized
@arguments("id")
def choose_infected(ids):
    """
    Returns an array of bools to select who gets infected
    """
    infection_pr = 0.3
    return np.random.random(len(ids)) <= INFECTION_PR


@vectorized
@arguments("id")
def choose_recovered(ids):
    """
    Returns an array of bools to select who recoveres
    """
    return np.random.random(len(ids)) <= RECOVERY_PR
