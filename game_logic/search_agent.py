from strategies.search.Deep_NEAT import DeepNEATCNN
from strategies.search.NNSearch import NNSearch
from strategies.search.NEAT_search import NEAT_search
from strategies.search.hunt_down import HuntDownStrategy
from strategies.search.probability import ProbabilisticStrategy
from strategies.search.random_strategy import RandomStrategy


class SearchAgent:
    def __init__(self, board_size, ship_sizes, strategy, weights=None, net=None):
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.board = [[0 for _ in range(self.board_size ** 2)] for _ in range(4)]
        self.strategy = self.init_strategy(strategy, weights, net)
        self.move_count = 0

    def init_strategy(self, strategy, weights, net):
        if strategy == "random":
            return RandomStrategy(self)
        elif strategy == "hunt_down":
            return HuntDownStrategy(self)
        elif strategy == "probabilistic":
            return ProbabilisticStrategy(self)
        elif strategy == "nn_search":
            return NNSearch(self, weights)
        elif strategy == "neat":
            return NEAT_search(self, net)

    def reset(self):
        self.board = [[0 for _ in range(self.board_size ** 2)] for _ in range(4)]
        self.move_count = 0
