from strategies.search.NEAT_Search import NEAT_search
from strategies.search.hunt_down import HuntDownStrategy
from strategies.search.probability import ProbabilisticStrategy
from strategies.search.random_strategy import RandomStrategy
from strategies.search.mcts import MCTSStrategy
from strategies.search.NNSearch import NNSearch


class SearchAgent:
    def __init__(
            self, board_size, strategy, net=None, optimizer="adam", lr=0.001, name=""
    ):
        self.board_size = board_size
        self.name = name
        self.strategy = self.init_strategy(strategy, net, optimizer, lr, name)

    def init_strategy(self, strategy, net, optimizer, lr, name):
        if strategy == "random":
            return RandomStrategy(self)
        elif strategy == "hunt_down":
            return HuntDownStrategy(self)
        elif strategy == "probabilistic":
            return ProbabilisticStrategy(self)
        elif strategy == "nn_search":
            return NNSearch(self, net, optimizer, lr)
        elif strategy == "neat":
            return NEAT_search(self, net)
        elif strategy == "mcts":
            return MCTSStrategy(self)
