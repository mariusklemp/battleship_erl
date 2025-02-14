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
        # Board tensor: shape (4, board_size, board_size)
        board_size = int((len(self.board[0])) ** 0.5)
        board_tensor = torch.tensor(self.board, dtype=torch.float32)

        # Reshape from (4, board_size*board_size) to (4, board_size, board_size)
        board_tensor = board_tensor.view(4, board_size, board_size)

        # Add batch dimension: (1, 4, board_size, board_size)
        board_tensor = board_tensor.unsqueeze(0)

        # Create extra features tensor
        ship_counts = Counter(self.remaining_ships)
        feature_vector = [ship_counts.get(i, 0) for i in range(1, 6)]
        extra_tensor = torch.tensor(feature_vector, dtype=torch.float32)

        return board_tensor, extra_tensor

    def state_tensor_canonical(self):
        """
        Returns the canonical board tensor, extra features tensor,
        and the number of 90° rotations applied.

        The board tensor is canonicalized by comparing all rotations.
        The extra features are left unchanged.
        """
        # Get the raw board tensor and reshape it to (4, board_size, board_size)
        board_size = int(
            (len(self.board[0])) ** 0.5
        )  # Calculate board size from flattened length
        board_tensor = torch.tensor(self.board, dtype=torch.float32)
        board_tensor = board_tensor.view(4, board_size, board_size)  # Reshape to 3D
        canonical_board, rotation = canonicalize_board(board_tensor)
        # Add the batch dimension: (1, 4, board_size, board_size)
        canonical_board = canonical_board.unsqueeze(0)

        # Extra features (ship counts remain unchanged)
        ship_counts = Counter(
            self.remaining_ships
        )  # Changed to use remaining_ships like state_tensor
        extra_features = [ship_counts.get(i, 0) for i in range(1, 6)]
        extra_tensor = torch.tensor(extra_features, dtype=torch.float32)

        return canonical_board, extra_tensor, rotation


def canonicalize_board(board: torch.Tensor):
    """
    Given a board tensor of shape (4, board_size, board_size),
    generate the 0°, 90°, 180°, and 270° rotations and choose the
    lexicographically smallest flattened version.

    Returns:
        canonical_board: torch.Tensor of shape (4, board_size, board_size)
        rotation: int (0, 1, 2, or 3) indicating how many 90° rotations were applied.
    """
    # Generate all 4 rotations. Note: torch.rot90 rotates along the last two dims.
    rotations = [torch.rot90(board, k=k, dims=(1, 2)) for k in range(4)]

    # Flatten each rotated board (across all channels) to compare lexicographically.
    flattened_versions = [rot.flatten().tolist() for rot in rotations]

    # Choose the rotation index with the smallest flattened representation.
    min_idx = min(range(4), key=lambda i: flattened_versions[i])

    return rotations[min_idx], min_idx
