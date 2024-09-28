from strategies.search.NNSearch import NNSearch
from strategies.search.hunt_down import HuntDownStrategy
from strategies.search.probability import ProbabilisticStrategy
from strategies.search.random_strategy import RandomStrategy


class SearchAgent:
    def __init__(self, board_size, ship_sizes, strategy, weights=None):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.search = [[0 for _ in range(self.board_size ** 2)] for _ in range(4)]
        self.strategy = self.init_strategy(strategy, weights)
        self.move_count = 0

    def init_strategy(self, strategy, weights):
        if strategy == "random":
            return RandomStrategy(self)
        elif strategy == "hunt_down":
            return HuntDownStrategy(self)
        elif strategy == "probabilistic":
            return ProbabilisticStrategy(self)
        elif strategy == "nn_search":
            return NNSearch(self, weights)

    def reset(self):
        self.search = [[0 for _ in range(self.board_size ** 2)] for _ in range(4)]
        self.move_count = 0
