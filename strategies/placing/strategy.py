from abc import abstractmethod, ABC


class Strategy(ABC):
    def __init__(self, placing_agent):
        self.placing_agent = placing_agent

    @abstractmethod
    def place_ships(self):
        pass
