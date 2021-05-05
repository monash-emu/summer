import random

import numpy as np
from typing import List, Callable, Any, Set, Dict
from .entities import BaseEntity


class Query:
    def __init__(self, entity_cls: BaseEntity, entity_values: Dict[str, np.ndarray], max_id: int):
        self._cls = entity_cls
        self._values = entity_values
        self._ids = range(max_id)

    def filter(self, **kwargs):
        # filter down ids with np.where
        self._ids = self._ids  # filtered down
        return self

    def write(self, enitites: List[BaseEntity]):
        pass

    def update(self, **kwargs):
        pass

    def choose(self, num_choices: int):

        random.choices(model._agents, k=3)

        return self

    def get(self, id: int) -> BaseEntity:
        pass

    def entities(self) -> List[BaseEntity]:
        return  # Entities as Python objects

    def ids(self) -> List[int]:
        return


"""

for agent in results:
    # Do stuff to agents
    agent.save()

# Or
sus_query = Query(**stuff).filter(disease=Disease.SUSCEPTIBLE).all()

.filter()
.random_choice()
.update()
.all() 


with sus_query as agents:
    for agent in agents:
        # Do stuff

"""