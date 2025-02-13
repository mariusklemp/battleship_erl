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

        ship_counts = Counter(self.remaining_ships)
        feature_vector = [ship_counts.get(i, 0) for i in range(1, 6)]
        extra_tensor = torch.tensor(feature_vector, dtype=torch.float32)

        return board_tensor, extra_tensor
