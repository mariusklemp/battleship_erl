import numpy as np
import torch

from strategies.search.strategy import Strategy
import torch.nn.functional as F


class NEAT_search(Strategy):
    def __init__(self, search_agent, net):
        super().__init__(search_agent)
        self.name = "NEAT_search"
        self.net = net

    def find_move(self, state):
        # Convert the board state (4 layers) into a tensor
        board_tensor = torch.tensor(state.board, dtype=torch.float32)

        # Reshape the board to have shape (batch_size, channels, height, width)
        board_size = self.search_agent.board_size
        board_tensor = board_tensor.view(1, 4, board_size, board_size)

        # Forward pass to get raw output (logits)
        output = self.net.forward(board_tensor).view(
            1, -1
        )  # Flatten the output to (1, board_size^2)

        # Get the 'unknown' layer (first layer) and flip it to mark known squares
        unknown_layer = torch.tensor(state.board[0], dtype=torch.float32).view(1, -1)

        # Mask the output: Set values where unknown_layer is 1 to -inf
        output[unknown_layer == 1] = -np.inf

        # Apply softmax to convert to probability distribution
        probabilities = F.softmax(output, dim=-1).squeeze(0)  # Shape: (board_size^2,)

        # Convert tensor to numpy array for random.choice
        probabilities_np = probabilities.detach().numpy()

        # Choose a move based on the probability distribution
        move = np.random.choice(self.search_agent.board_size**2, p=probabilities_np)

        return move

    # def find_move(self, state):
    #     # Flatten the 4-layer 10x10 board to a single list of 400 inputs
    #     input_data = [cell for layer in state.board for cell in layer]

    #     # Activate the network with the flattened input to get output
    #     output = np.array(self.net.activate(input_data))

    #     # Mask the output to ensure it doesnt choose a square that has already been chosen
    #     unknown_layer = np.array(state.board[0]).flatten()
    #     output[unknown_layer == 1] = -np.inf
    #     exp_output = np.exp(output - np.max(output))
    #     probabilities = exp_output / exp_output.sum()

    #     # Choose a move based on the probability distribution
    #     move = np.random.choice(len(output), p=probabilities)

    #     return move
