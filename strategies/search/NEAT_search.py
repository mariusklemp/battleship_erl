import random
import numpy as np
from strategies.search.strategy import Strategy


class NEAT_search(Strategy):
    def __init__(self, search_agent, net):
        super().__init__(search_agent)
        self.name = "NEAT_search"
        self.net = net

    def find_move(self, state):
        # Flatten the 4-layer 10x10 board to a single list of 400 inputs
        input_data = [cell for layer in state.board for cell in layer]

        # Activate the network with the flattened input to get output
        output = np.array(self.net.activate(input_data))

        # Mask the output to ensure it doesnt choose a square that has already been chosen
        unknown_layer = np.array(state.board[0]).flatten()
        output[unknown_layer == 1] = -np.inf
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()

        # Choose a move based on the probability distribution
        move = np.random.choice(len(output), p=probabilities)

        return move
