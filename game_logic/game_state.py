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

        # Create ship matrix representation
        ship_counts = Counter(self.remaining_ships)
        ship_matrix = torch.zeros((1, board_size, board_size), dtype=torch.float32)
        
        # Fill the matrix with ships - each column represents a ship size
        for size in range(1, board_size + 1):
            count = ship_counts.get(size, 0)
            for i in range(count):
                ship_matrix[0, i, size-1] = 1

        # Concatenate ship matrix as the 5th channel
        board_tensor = torch.cat([board_tensor, ship_matrix], dim=0)
        
        # Add batch dimension: (1, 5, board_size, board_size)
        board_tensor = board_tensor.unsqueeze(0)

        # Return the board tensor (with ship matrix) and None for extra features
        return board_tensor

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

        # Create ship matrix representation
        ship_counts = Counter(self.remaining_ships)
        ship_matrix = torch.zeros((board_size, board_size), dtype=torch.float32)
        
        # Fill the matrix with ships - each column represents a ship size
        for size in range(1, board_size + 1):
            count = ship_counts.get(size, 0)
            for i in range(count):
                ship_matrix[i, size-1] = 1

        # Add batch dimension: (1, board_size, board_size)
        ship_matrix = ship_matrix.unsqueeze(0)

        return canonical_board, ship_matrix, rotation


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
