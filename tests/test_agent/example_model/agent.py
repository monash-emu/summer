from summer.agent import fields
from summer.agent.entities import BaseAgent

from .constants import Disease, NO_DATE


class Disease:
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3
    DEAD = 4


class Agent(BaseAgent):
    """
    An individual who is a part of a tribe and/or trade network.
    """

    death_date = fields.IntegerField(default=NO_DATE)
    recovery_date = fields.IntegerField(default=NO_DATE)
    disease = fields.IntegerField(default=Disease.SUSCEPTIBLE)
