from strategies.search.NNSearch import NNSearch
from colorama import Fore, Style

from strategies.search.NEAT_search import NEAT_search
from strategies.search.hunt_down import HuntDownStrategy
from strategies.search.probability import ProbabilisticStrategy
from strategies.search.random_strategy import RandomStrategy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class SearchAgent:
    def __init__(
        self, board_size, ship_sizes, strategy, net=None, optimizer="adam", lr=0.001
    ):
        self.board_size = board_size
        # self.ship_sizes = ship_sizes
        # self.board = [[0 for _ in range(self.board_size ** 2)] for _ in range(4)]
        self.strategy = self.init_strategy(strategy, net, optimizer, lr)
        self.move_count = 0

    def init_strategy(self, strategy, net, optimizer, lr):
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

    def reset(self):
        self.board = [[0 for _ in range(self.board_size**2)] for _ in range(4)]
        self.move_count = 0

    def print_board(self):
        board_state = self.board

        for row in range(self.board_size):
            row_str = ""
            for col in range(self.board_size):
                # Check if the ship is sunk
                if board_state[3][row * self.board_size + col] == 1:
                    row_str += f"{Fore.RED} S {Style.RESET_ALL}| "  # Sunk
                # If not sunk, check hit and miss layers
                elif board_state[1][row * self.board_size + col] == 1:
                    row_str += f"{Fore.YELLOW} X {Style.RESET_ALL}| "  # Hit
                elif board_state[2][row * self.board_size + col] == 1:
                    row_str += f"{Fore.BLUE} - {Style.RESET_ALL}| "  # Miss
                else:
                    row_str += f"   | "  # Unknown (default)

            print(row_str)
            print("-" * (self.board_size * 4))  # Add separators
