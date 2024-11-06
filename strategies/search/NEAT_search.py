import neat

from strategies.search.strategy import Strategy


class NEAT_search(Strategy):
    def __init__(self, search_agent, net):
        self.name = "NEAT_search"
        self.search_agent = search_agent
        self.neat = net

    def find_move(self):
        output = self.neat.activate(self.search_agent.search)
