import numpy as np

from summer.agent import fields
from summer.agent.entities import BaseNetwork

from .constants import NetworkType

MEAN_TRIBE_SIZE = 12


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


class Network(BaseNetwork):
    """
    A tribe of individuals
    """

    type = fields.IntegerField(default=NetworkType.TRIBE)
    capacity = fields.IntegerField(distribution=poisson_with_floor(MEAN_TRIBE_SIZE, 2))
