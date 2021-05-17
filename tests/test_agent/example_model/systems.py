import random

import numpy as np

from summer.agent.registry import arguments, vectorized
from summer.agent.model import AgentModel

from .constants import NetworkType, Disease
from .network import Network

INTIIAL_INFECTED = 10
MAX_RECOVERY_DAYS = 5
MAX_DEATH_DAYS = 4
RECOVERY_PR = 0.7
INFECTION_PR = 0.4


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


def setup_trade_networks(model: AgentModel):
    """
    Sets up trade networks of people from tribes
    """
    # Randomly add some inter tribe connections - "trade connections"
    # Each model is has 2 trade networks with 2 other random tribes
    for src_tribe in model.networks.query.filter(type=NetworkType.TRIBE).all():
        dest_tribes = (
            model.networks.query.deselect([src_tribe.id])
            .filter(type=NetworkType.TRIBE)
            .choose(2)
            .all()
        )
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
        infectees_query = model.agents.query.choose(initial_infected)
        apply_infection(infectees_query, time=0)

    return seed_infection


def apply_infection(infectees_query, time):
    # Select some infected to recover.
    recovered_query = infectees_query.where(choose_recovered)
    recovered_query.update(disease=Disease.INFECTED, recovery_date=build_choose_recovery_date(time))
    # The remainder die.
    infectees_query.deselect(recovered_query.ids()).update(
        disease=Disease.INFECTED, death_date=build_choose_death_date(time)
    )


def recovery_system(model: AgentModel, time: float):
    # Everyone who recovers by this date should be set to recovered
    model.agents.query.filter(
        disease=Disease.INFECTED, recovery_date=time, recovery_date__gte=0
    ).update(disease=Disease.RECOVERED)


def death_system(model: AgentModel, time: float):
    # Everyone who dies by this date should be set to dead
    model.agents.query.filter(disease=Disease.INFECTED, death_date=time).update(
        disease=Disease.DEAD
    )


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
            apply_infection(inf_contacts_query, time)


@vectorized
@arguments("id")
def choose_recovered(ids):
    """
    Returns an array of bools to select who recoveres (as opposed to dying).
    """
    return np.random.random(len(ids)) <= RECOVERY_PR


def build_choose_recovery_date(time):
    @vectorized
    @arguments("id")
    def choose_recovery_date(ids):
        """
        A person randomly recovers 0-MAX_RECOVERY_DAYS days after they get infected.
        """
        return time + np.round(np.random.random(len(ids)) * MAX_RECOVERY_DAYS)

    return choose_recovery_date


def build_choose_death_date(time):
    @vectorized
    @arguments("id")
    def choose_death_date(ids):
        """
        A person randomly dies 0-MAX_DEATH_DAYS days after they get infected.
        """
        return time + np.round(np.random.random(len(ids)) * MAX_DEATH_DAYS)

    return choose_death_date


@vectorized
@arguments("id")
def choose_infected(ids):
    """
    Returns an array of bools to select who gets infected
    """
    return np.random.random(len(ids)) <= INFECTION_PR
