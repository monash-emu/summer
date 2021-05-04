from abc import ABC


class BaseAgent(ABC):
    def __init__(self, agent_id: int):
        self.id = agent_id

    def __str__(self):
        return f"<BaseAgent {self.id}>"