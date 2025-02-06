import torch
from collections import Counter


class GameState:
    def __init__(self, board, move_count, placing, remaining_ships):
        self.board = board
        self.move_count = move_count
        self.placing = placing
        self.remaining_ships = remaining_ships

    def state_tensor(self):
        """Returns both the board tensor and extra features tensor."""
        # Board tensor: shape (1, 4, board_size, board_size)
        board_tensor = torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)

        # Count ships by size
        ship_size_counts = Counter(ship.size for ship in self.placing.ships)

        # Create a list of 6 elements (for ships of size 1-6)
        # Each element represents how many ships of that size exist
        extra_features = [ship_size_counts.get(i, 0) for i in range(1, 7)]
        extra_tensor = torch.tensor(extra_features, dtype=torch.float32)

        return board_tensor, extra_tensor
