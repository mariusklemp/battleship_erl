from abc import abstractmethod, ABC


class Strategy(ABC):
    def __init__(self, search_agent):
        self.search_agent = search_agent

    @abstractmethod
    def find_move(self, state):
        pass
