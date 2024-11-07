from game_logic.ship import Ship

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNPlacing(nn.Module):
    """
    This does not work yet. Its doesnt place valid ships.
    """

    def __init__(self, placing_agent):
        super(NNPlacing, self).__init__()
        self.placing_agent = placing_agent

        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers for output
        self.fc1 = nn.Linear(64 * self.placing_agent.board_size * self.placing_agent.board_size, 128)
        self.fc2 = nn.Linear(128, 3)  # Outputs: (x, y, direction)

    def forward(self, board_tensor):
        # Input shape: (batch_size, 1, board_size, board_size)
        x = F.relu(self.conv1(board_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the 2D convolutional features
        x = x.view(x.size(0), -1)  # Flatten the output of conv3

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))

        # Output (x, y, direction)
        out = self.fc2(x)

        # Return the network output
        return out

    def place_ships(self):
        """
        This method places ships on the board using the neural network.
        """
        # Convert the board to a PyTorch tensor with shape (1, 1, board_size, board_size)
        board_tensor = torch.Tensor(self.placing_agent.board).unsqueeze(0).unsqueeze(0)

        # Get placements for each ship from the network
        for size in self.placing_agent.ship_sizes:
            placed = False

            while not placed:
                # Forward pass to get the ship placement
                output = self.forward(board_tensor)

                # Output (x, y, direction)
                x, y, direction = output[0, 0], output[0, 1], output[0, 2]

                # Convert to valid board coordinates and direction
                x = int(x.item() % self.placing_agent.board_size)
                y = int(y.item() % self.placing_agent.board_size)
                direction = int(direction.item() % 2)  # 0: horizontal, 1: vertical

                # Adjust the placement to ensure validity
                x, y = self.adjust_to_valid_position(x, y, direction, size)

                print(f"Placing ship of size {size} at ({x}, {y}) in direction {direction}")
                # Create a new ship object with calculated placement
                ship = Ship(size, self.placing_agent.board_size, x, y, direction)

                # Check if the placement is valid
                if self.placing_agent.check_valid_placement(ship):
                    print("Valid placement")
                    self.placing_agent.ships.append(ship)

                    # Update the board with the placed ship
                    self.placing_agent.update_board(x, y, size, direction)
                    placed = True
                else:
                    print("Not valid placement")

    def adjust_to_valid_position(self, x, y, direction, size):
        """
        Adjust the ship placement if it goes out of bounds or overlaps.
        """

        row = x
        col = y

        # Check if the ship would go out of bounds and adjust accordingly
        if direction == 0:  # Horizontal
            while col + size > self.placing_agent.board_size:
                col = (col + 1) % self.placing_agent.board_size  # Move left to fit
        elif direction == 1:  # Vertical
            while row + size > self.placing_agent.board_size:
                row = (row + 1) % self.placing_agent.board_size  # Move up to fit

        # You can add more logic here to check for overlaps and make further adjustments if needed

        print(f"Adjusted position: ({row}, {col})")
        return row, col
