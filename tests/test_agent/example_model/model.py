from summer.agent.model import AgentModel

from . import systems
from .agent import Agent
from .network import Network


def build_example_model(
    num_people: int,
    initial_infected: int,
    end_time: int = 10,
):
    model = AgentModel(start_time=0, end_time=end_time, timestep=1)
    # Add model entities.
    model.set_agent_class(Agent, initial_number=num_people)
    model.set_network_class(Network)

    # Define model setup.
    model.add_setup_step(systems.setup_tribes)
    model.add_setup_step(systems.setup_trade_networks)
    model.add_setup_step(systems.get_seed_infection(initial_infected))

    # Define model runtime.
    model.add_runtime_step(systems.recovery_system)
    model.add_runtime_step(systems.death_system)
    model.add_runtime_step(systems.infection_system)
    return model
