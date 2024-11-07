import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from strategies.search.strategy import Strategy


class NNSearch(nn.Module, Strategy):
    def __init__(self, search_agent, weights=None):
        super().__init__()
        self.name = "nn_search"
        self.search_agent = search_agent

        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers for output
        self.fc1 = nn.Linear(64 * self.search_agent.board_size * self.search_agent.board_size, 128)
        self.fc2 = nn.Linear(128, self.search_agent.board_size ** 2)  # Outputs: distribution over all moves

        # Load weights if provided
        if weights is not None:
            self.load_state_dict(weights)  # Directly load the state_dict

    def forward(self, board_tensor):
        # Pass through CNN layers with ReLU activation
        x = F.relu(self.conv1(board_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the convolutional features
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out

    def find_move(self):
        self.search_agent.move_count += 1

        # Convert the board state (4 layers) into a tensor
        board_tensor = torch.tensor(self.search_agent.board, dtype=torch.float32)

        # Reshape the board to have shape (batch_size, channels, height, width)
        # For example, (1, 4, 10, 10) for a 10x10 board
        board_size = self.search_agent.board_size
        board_tensor = board_tensor.view(1, 4, board_size, board_size)

        # Forward pass to get raw output (logits)
        output = self.forward(board_tensor).view(1, -1)  # Flatten the output to (1, board_size^2)

        # Get the 'unknown' layer (first layer) and flip it to mark known squares
        unknown_layer = torch.tensor(self.search_agent.board[0], dtype=torch.float32).view(1, -1)

        # Mask the output: Set values where unknown_layer is 1 to -inf
        output[unknown_layer == 1] = float('-inf')

        # Apply softmax to convert to probability distribution
        probabilities = F.softmax(output, dim=-1).squeeze(0)  # Shape: (board_size^2,)

        # Convert tensor to numpy array for random.choice
        probabilities_np = probabilities.detach().numpy()

        # Choose a move based on the probability distribution
        move = np.random.choice(self.search_agent.board_size ** 2, p=probabilities_np)

        return move
