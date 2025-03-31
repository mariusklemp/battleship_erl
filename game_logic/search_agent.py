from strategies.search.hunt_down import HuntDownStrategy
from strategies.search.random import RandomStrategy
from strategies.search.mcts import MCTSStrategy
from strategies.search.nn_search import NNSearch


class SearchAgent:
    def __init__(
            self, board_size, strategy, net=None, optimizer="adam", lr=0.001, name=""
    ):
        self.board_size = board_size
        self.name = name
        self.strategy = self.init_strategy(strategy, net, optimizer, lr)

    def init_strategy(self, strategy, net, optimizer, lr):
        if strategy == "random":
            return RandomStrategy(self)
        elif strategy == "hunt_down":
            return HuntDownStrategy(self)
        elif strategy == "nn_search":
            return NNSearch(self, net, optimizer, lr)
        elif strategy == "mcts":
            return MCTSStrategy(self)
        else:
            raise ValueError("Unknown search strategy")
